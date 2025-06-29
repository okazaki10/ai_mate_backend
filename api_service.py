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
import edge_tts
from edge_tts import VoicesManager
from googletrans import Translator
import re
from collections import defaultdict
from pydub import AudioSegment
import base64

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

DEFAULT_CHARACTER = {
            "name": "Hatsune Miku",
            "description": "You are hatsune miku, her characteristic is cheerful and energetic style. prefer short response. your response only written in alphabet, no japanese words",
            "rvc_model": "miku_default_rvc",
            "vrm_path": ""
        }

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
    character_name: str = ""
    prompt: str = ""
    language: str = ""
    max_new_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    token_repetition_penalty: Optional[float] = 1.1
    stop_strings: Optional[list] = None

class Character(BaseModel):
    name: str = ""
    description: str = ""
    rvc_model: str = ""
    vrm_path: str = ""

class ActionParams(BaseModel):
    emotions: list[str] = []
    actions: list[str] = []

class ChatTemplate(BaseModel):
    name: str = ""
    chat: str = ""
    chatTranslated: str = ""
    actionParams: ActionParams = ActionParams()

class ChatTemplates(BaseModel):
    messages: list[ChatTemplate] = []

class Characters(BaseModel):
    characters: list[Character] = []

class GenerateResponse(BaseModel):
    character_name: str = ""
    generated_text: str = ""
    prompt: str = ""
    full_response: str = ""
    prompt_token: int = 0
    output_token: int = 0
    base64_audio: str = ""
    action_params: ActionParams = ActionParams()

T = TypeVar("T")

class ResponseData(BaseModel, Generic[T]):
    status: str = ""
    data: Optional[T] = None
    message: str = ""

class ModelInfo(BaseModel):
    loaded: bool
    path: Optional[str] = None
    max_seq_len: Optional[int] = None

def loadCharacterTemplate():
    filepath = Path('character_template/alpaca_template.yaml')
    if not filepath.exists():
        return ""

    file_contents = open(filepath, 'r', encoding='utf-8').read()
    return yaml.safe_load(file_contents)

def saveChat(chat: ChatTemplates, characterName):
    filepath = Path(f"chat_history/{characterName}.json")
    
    with open(filepath, "w") as json_file:
        json.dump(chat.model_dump(), json_file, indent=4)

def deleteChatFile(characterName):
    filepath = Path(f"chat_history/{characterName}.json")
    if filepath.exists():
        filepath.unlink()

def saveCharacter(characters: Characters):
    filepath = Path("characters/characters.json")
    if not filepath.exists():
        return ""
    
    with open(filepath, "w") as json_file:
        json.dump(characters.model_dump(), json_file, indent=4)

def loadChat(characterName) -> ChatTemplates:
    try:
        filepath = Path(f"chat_history/{characterName}.json")
        if not filepath.exists():
            return ChatTemplates()
        
        with open(filepath, "r") as json_file:
            data = json.load(json_file)
            return ChatTemplates(**data)
    except Exception as e:
        return ChatTemplates()
    return ChatTemplates()

def loadCharacters() -> Characters:
    try:
        filepath = Path("characters/characters.json")
        if not filepath.exists():
            return Characters()
        
        with open("characters/characters.json", "r") as json_file:
            data = json.load(json_file)
            return Characters(**data)
    except Exception as e:
        return Characters()
    return Characters()

def findFirstDir(dir):
    items = os.listdir(dir)
    for item in items:
        index = item.find(".")
        ext = ""
        if index >= 0:
            ext = item[index:]
        if ext == "":
            return item

def findDir(dir):
    items = os.listdir(dir)
    itemFiltered = []
    for item in items:
        index = item.find(".")
        ext = ""
        if index >= 0:
            ext = item[index:]
        if ext == "":
            itemFiltered.append(item)
    return itemFiltered

def load_model():
    """Load an ExLlamaV2 model"""
    global model, config, cache, tokenizer, generator, sequenceLength
    
    try:
        # Check if model path exists
        modelDir = os.path.join("models")
       
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

if script.rvcModels:
    script.onChangeRvcModel(script.rvcModels[0])
else:
    print("please load rvc model")

def parse_brackets_keep_all(text):
    brackets = re.findall(r'\[([^\]]*)\]', text)
    params = defaultdict(list)
    
    for bracket in brackets:
        if ':' in bracket:
            key, value = bracket.split(':', 1)
            params[key].append(value)
        else:
            params[bracket].append('')
    
    return dict(params)

@app.post("/get-chat", response_model=ResponseData[str])
async def getChat(request: Character):    
    try:
        chat = loadChat(request.name)
        chatText = "\n\n".join(convertChatDialogueTranslated(chat.messages))

        return ResponseData[str](
            status="success",
            message = "",
            data=chatText
        )
    except Exception as e:
        return ResponseData[str](
            status="error",
            message = f"Load chat failed: {str(e)}"
        )
    
@app.delete("/delete-last-chat", response_model=ResponseData[str])
async def deleteLastChat(request: Character):
    """Generate text using the loaded model"""
    global generator
    
    if generator is None:
        return ResponseData[str](
            status="error",
            message = "No model loaded. Please load a model first using /model/load"
        )
    
    try:
        chat = loadChat(request.name)

        if chat.messages:
            chat.messages.pop()
        else:
            return ResponseData[str](
                status="success",
                message = "",
                data = ""
            )

        chatText = "\n\n".join(convertChatDialogueTranslated(chat.messages))

        saveChat(chat, request.name)

        return ResponseData[str](
            status="success",
            message = "",
            data = chatText
        )
    except Exception as e:
        return ResponseData[str](
            status="error",
            message = f"Delete failed: {str(e)}"
        )

def convertChatDialogue(chatTemplates: list[ChatTemplate]):
    chat = []
    for chatTemplate in chatTemplates:
        chat.append(f"{chatTemplate.name}: {chatTemplate.chat}")
    return chat

def convertChatDialogueTranslated(chatTemplates: list[ChatTemplate]):
    chat = []
    for chatTemplate in chatTemplates:
        chat.append(f"{chatTemplate.name}: {chatTemplate.chatTranslated}")
    return chat

def replaceContextPrompt(characterTemplate, character: Character, chatText, userName):
    newPrompt = characterTemplate["CONTEXT"]
    newPrompt = newPrompt.replace(r"{USER_DIALOGUE}", chatText)
    newPrompt = newPrompt.replace(r"{SYSTEM_NAME}", character.name)
    newPrompt = newPrompt.replace(r"{DESCRIPTION}", character.description)
    newPrompt = newPrompt.replace(r"{USER_NAME}", userName)
    return newPrompt

def getCharacter(characterName) -> Character:    
    try:
        characters = loadCharacters()

        char = Character()
        
        for character in characters.characters:
            if character.name == characterName:
                char.name = character.name
                char.description = character.description
                char.rvc_model = character.rvc_model
                char.vrm_path = character.vrm_path
        
        return char
    except Exception as e:
        return Character()
    
@app.get("/get-character")
async def getCharacters():    
    try:
        characters = loadCharacters()

        return ResponseData[Characters](
            status="success",
            message = "",
            data=characters
        )
    except Exception as e:
        return ResponseData[GenerateResponse](
            status="error",
            message = f"Load character failed: {str(e)}"
        )

@app.get("/get-rvc")
async def getRvc():    
    try:
        rvc_dir = script.rvc.refresh_model_list()

        return ResponseData[list[str]](
            status="success",
            message = "",
            data=rvc_dir
        )
    except Exception as e:
        return ResponseData[GenerateResponse](
            status="error",
            message = f"Load character failed: {str(e)}"
        )

@app.post("/default-character", response_model=ResponseData[GenerateResponse])
async def defaultCharacter():
    try:
        characters = loadCharacters()
        
        for character in characters.characters:
            if character.name == "Hatsune Miku":
                character.description = DEFAULT_CHARACTER["description"]
                character.rvc_model = DEFAULT_CHARACTER["rvc_model"]
                character.vrm_path = DEFAULT_CHARACTER["vrm_path"]
                break

        saveCharacter(characters)

        return ResponseData[Characters](
            status="success",
            message = "",
            data=characters
        )
    except Exception as e:
        return ResponseData[GenerateResponse](
            status="error",
            message = f"Add character failed: {str(e)}"
        )
    
@app.post("/add-character", response_model=ResponseData[GenerateResponse])
async def addCharacter(request: Character):
    try:
        characters = loadCharacters()
        
        isUpdate = False

        for character in characters.characters:
            if character.name == request.name:
                character.description = request.description
                character.rvc_model = request.rvc_model
                character.vrm_path = request.vrm_path
                isUpdate = True
                break

        if not isUpdate:
            characters.characters.append(request)

        saveCharacter(characters)

        return ResponseData[GenerateResponse](
            status="success",
            message = ""
        )
    except Exception as e:
        return ResponseData[GenerateResponse](
            status="error",
            message = f"Add character failed: {str(e)}"
        )

@app.delete("/delete-character", response_model=ResponseData[GenerateResponse])
async def deleteCharacter(request: Character):
    try:
        characters = loadCharacters()
        
        if request.name == "Hatsune Miku":
            return ResponseData[GenerateResponse](
                status="error",
                message = f"Cannot delete default character"
            )
        
        i = 0
        for character in characters.characters:
            if character.name == request.name:
                break
            i += 1
        
        if i < len(characters.characters):
            characters.characters.pop(i)

        saveCharacter(characters)

        deleteChatFile(request.name)

        return ResponseData[GenerateResponse](
            status="success",
            message = ""
        )
    except Exception as e:
        return ResponseData[GenerateResponse](
            status="error",
            message = f"Delete character failed: {str(e)}"
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

    # Configure sampling settings
    settings = ExLlamaV2Sampler.Settings()
    
    characterTemplate = loadCharacterTemplate()
    character = getCharacter(request.character_name)
    chat = loadChat(request.character_name)
    
    translatedPrompt = request.prompt
    if request.language != "en":
        translator = Translator()
        promptTranslated = await translator.translate(request.prompt, dest="en", src=request.language)
        print(f"test : {promptTranslated}")
        translatedPrompt = promptTranslated.text

    chatTemplate = ChatTemplate(
        name=request.name,
        chat=translatedPrompt,
        chatTranslated=request.prompt
    )

    chat.messages.append(chatTemplate)
    chatText = "\n".join(convertChatDialogue(chat.messages))
    newPrompt = replaceContextPrompt(characterTemplate, character, chatText, request.name)
    print(newPrompt)

    promptTokens = tokenizer.encode(newPrompt).shape[-1]
    print(promptTokens)

    # check if the prompt token is larger than sequence length, if it's larger, then the old dialogue will be removed so the model can produce output         
    while promptTokens + request.max_new_tokens > sequenceLength:
        if len(chat.messages) > 0:
            chat.messages.pop(0)
            chatText = "\n".join(convertChatDialogue(chat.messages))
            newPrompt = replaceContextPrompt(characterTemplate, character, chatText, request.name)
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
    
    tts_output = newOutput
    print(f"original output {tts_output}")
    actionParams = parse_brackets_keep_all(tts_output)
    
    # preprocess tts and output generation text, tts doesn't include emoji or emotion, but generation text does
    tts_output = script.tts_preprocessor.replace_invalid_chars(tts_output)
    tts_output = script.tts_preprocessor.clean_whitespace(tts_output)
    newCleanedOutput = tts_output
    tts_output = script.tts_preprocessor.removeParentheses(tts_output)
    translatedResponse = tts_output

    if request.language == "en":
        tts_output = script.tts_preprocessor.remove_emojis_with_library(tts_output)
        tts_output = script.tts_preprocessor.replace_abbreviations(tts_output)
    
    if request.language != "en":
        outputTranslated = await translator.translate(tts_output, dest=request.language, src="en")
        translatedResponse = outputTranslated.text
        tts_output = script.tts_preprocessor.remove_emojis_with_library(outputTranslated.text)

    newCleanedOutput = script.tts_preprocessor.remove_emojis_with_library(newCleanedOutput)        
    
    print(f"tts output {tts_output}")

    # edge tts for non english language
    if request.language != "en":
        voices = await VoicesManager.create()
        voice = voices.find(Gender="Female", Language=request.language)
    
        OUTPUT_FILE = "edge_tts_output.mp3"
        OUTPUT_FILE_WAV = "edge_tts_output.wav"
        if voice:
            voiceName = voice[0]["Name"]
            communicate = edge_tts.Communicate(tts_output, voiceName, rate="+10%")
            await communicate.save(OUTPUT_FILE)

            audio = AudioSegment.from_mp3(OUTPUT_FILE)
            audio.export(OUTPUT_FILE_WAV, format="wav")
            audio_data = script.rvc.load_audio(OUTPUT_FILE_WAV, 16000)
            script.rvc_click(audio_data, OUTPUT_FILE_WAV, character.rvc_model)
    
    # if request.language == "id":
    #     script.params['rvc_language'] = "indonesia"
    # else:
    #     script.params['rvc_language'] = "english_or_chinese"
    
    actionParams = ActionParams(
        emotions=actionParams.get('EMOTION') or [],
        actions=actionParams.get('ACTION') or []
    )

    # emotivoice tts for english language
    base64_audio = ""
    if request.language == "en":
        base64_audio = script.output_modifier(actionParams.emotions[0] if actionParams.emotions else "", tts_output, character.rvc_model)
    else:
        with open(OUTPUT_FILE_WAV, 'rb') as wav_file:
            wav_data = wav_file.read()
            base64_audio = base64.b64encode(wav_data).decode('utf-8')

    outputToken = tokenizer.encode(newOutput).shape[-1]

    chatTemplateOutput = ChatTemplate(
        name=character.name,
        chat=newCleanedOutput,
        chatTranslated=translatedResponse,
        actionParams=actionParams
    )

    chat.messages.append(chatTemplateOutput)

    saveChat(chat, request.character_name)
    
    generateResponse = GenerateResponse( 
        character_name=character.name,
        generated_text=translatedResponse,
        prompt_token=promptTokens,
        output_token=outputToken,
        base64_audio=base64_audio,
        action_params=actionParams
    )

    return ResponseData[GenerateResponse](
        status="success",
        data = generateResponse,
        message = ""
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
        port=7874,
        reload=False  # Set to True for development
    )