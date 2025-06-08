from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from typing import Generic, TypeVar
from Emotivoice_RVC_TTS import script
import uvicorn
import os
import yaml
import json

# ExLlamaV2 imports
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

app = FastAPI(title="ExLlamaV2 API", description="REST API for ExLlamaV2 text generation")

# Global variables to store the model components
model = None
config = None
cache = None
tokenizer = None
generator = None
sequenceLength = None

class ModelLoadRequest(BaseModel):
    path: str
    max_seq_len: Optional[int] = 8000
    cache_max_seq_len: Optional[int] = 8000

class GenerateRequest(BaseModel):
    name: str = ""
    prompt: str = ""
    max_new_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    token_repetition_penalty: Optional[float] = 1.1
    stop_strings: Optional[list] = None

class DeleteRequest(BaseModel):
    index: int = 0

class GenerateResponse(BaseModel):
    generated_text: str = ""
    prompt: str = ""
    full_response: str = ""
    prompt_token: int = 0
    output_token: int = 0

T = TypeVar("T")

class ResponseData(BaseModel, Generic[T]):
    status: str = ""
    data: Optional[T] = None
    message: str = ""

class ModelInfo(BaseModel):
    loaded: bool
    path: Optional[str] = None
    max_seq_len: Optional[int] = None

def loadCharacter():
    filepath = Path('character/character.yaml')
    if not filepath.exists():
        return ""

    file_contents = open(filepath, 'r', encoding='utf-8').read()
    return yaml.safe_load(file_contents)

def saveChat(chatMap):
    with open("chat_history/chat.json", "w") as json_file:
        json.dump(chatMap, json_file, indent=4)

def loadChat():
    with open("chat_history/chat.json", "r") as json_file:
        return json.load(json_file)
    return ""

def findFirstDir(dir):
    items = os.listdir(dir)
    for item in items:
        index = item.find(".")
        ext = ""
        if index >= 0:
            ext = item[index:]
        if ext == "":
            if item == "silicon-maid-7b":
                return item

def load_model():
    """Load an ExLlamaV2 model"""
    global model, config, cache, tokenizer, generator, sequenceLength
    
    try:
        # Check if model path exists
        modelDir = os.path.join("..", "..", "user_data", "models")
       
        modelPath = os.path.join(modelDir,findFirstDir(modelDir))

        if not os.path.exists(modelPath):
            raise HTTPException(status_code=400, detail=f"Model path does not exist: {modelPath}")
        
        # Initialize config
        config = ExLlamaV2Config()
        config.model_dir = modelPath
        config.prepare()
        
        sequenceLength = 4096
        # Set sequence length
        config.max_seq_len = sequenceLength
        
        # Initialize model
        model = ExLlamaV2(config)
        print("Loading model...")
        model.load()
        
        # Initialize tokenizer
        tokenizer = ExLlamaV2Tokenizer(config)
        
        # Initialize cache
        cache = ExLlamaV2Cache(model, max_seq_len=sequenceLength)
        
        # Initialize generator
        generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)
        
        print(f"Model loaded successfully {modelPath} {sequenceLength}")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

load_model()
script.setup()

@app.post("/delete-chat", response_model=ResponseData[GenerateResponse])
async def deleteChat(request: DeleteRequest):
    """Generate text using the loaded model"""
    global generator
    
    if generator is None:
        return ResponseData[GenerateResponse](
            status="error",
            message = "No model loaded. Please load a model first using /model/load"
        )
    
    try:
        chat = loadChat()

        chat['chat'].pop(request.index)

        saveChat(chat)

        return ResponseData[GenerateResponse](
            status="success",
            message = ""
        )
    except Exception as e:
        return ResponseData[GenerateResponse](
            status="error",
            message = f"Delete failed: {str(e)}"
        )
    
@app.post("/generate", response_model=ResponseData[GenerateResponse])
async def generate_text(request: GenerateRequest):
    """Generate text using the loaded model"""
    global generator
    
    if generator is None:
        return ResponseData[GenerateResponse](
            status="error",
            message = "No model loaded. Please load a model first using /model/load"
        )
    
    try:
        # Configure sampling settings
        settings = ExLlamaV2Sampler.Settings()
        
        character = loadCharacter()
        chat = loadChat()

        chat['chat'].append(f"{request.name}: {request.prompt}")
        chatText = "\n".join(chat['chat'])
        newPrompt = character["context"].replace(r"{USER_DIALOGUE}", chatText)
        print(newPrompt)

        promptTokens = tokenizer.encode(newPrompt).shape[-1]
        print(promptTokens)

        # check if the prompt token is larger than sequence length         
        while promptTokens + request.max_new_tokens > sequenceLength:
            if len(chat['chat']) > 0:
                chat['chat'].pop(0)
                chatText = "\n".join(chat['chat'])
                newPrompt = character["context"].replace(r"{USER_DIALOGUE}", chatText)
                promptTokens = tokenizer.encode(newPrompt).shape[-1]
            else:
                break

        # Generate text
        output = generator.generate_simple(
            newPrompt,
            settings,
            request.max_new_tokens,
            seed=None
        )

        newOutput = output[len(newPrompt):].strip()

        if newOutput == "":
            return ResponseData[GenerateResponse](
                status="error",
                message = "context is larger than sequence length, please increase the sequence length or decrease the context or decrease the chat prompt"
            )
        
    
        script.output_modifier(newOutput)

        outputToken = tokenizer.encode(newOutput).shape[-1]

        chat['chat'].append(f"{character['name']}: {newOutput}")

        saveChat(chat)
        
        generateResponse = GenerateResponse(
            generated_text=newOutput,
            prompt_token=promptTokens,
            output_token=outputToken
        )

        return ResponseData[GenerateResponse](
            status="success",
            data = generateResponse,
            message = ""
        )
    except Exception as e:
        return ResponseData[GenerateResponse](
            status="error",
            message = f"Generation failed: {str(e)}"
        )

@app.post("/model/unload")
async def unload_model():
    """Unload the current model to free memory"""
    global model, config, cache, tokenizer, generator
    
    try:
        # Clean up model components
        if generator:
            del generator
            generator = None
        if cache:
            del cache
            cache = None
        if tokenizer:
            del tokenizer
            tokenizer = None
        if model:
            del model
            model = None
        if config:
            del config
            config = None
            
        # Force garbage collection
        import gc
        gc.collect()
        
        return {"message": "Model unloaded successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to unload model: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,  # Pass the app directly instead of module string
        host="0.0.0.0",
        port=8000,
        reload=False  # Set to True for development
    )