from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
from typing import Generic, TypeVar
import uvicorn
import os
import yaml
import json
import edge_tts
from edge_tts import VoicesManager
import re
from collections import defaultdict
from pydub import AudioSegment
import base64
import logging
from llama_cpp import Llama
from url_safe_translator import URLSafeTranslator
import librosa
import argparse
import subprocess
import sys
from web_search import WebSearchLLM

# Save original argv
original_argv = sys.argv[:]

# Remove problematic arguments before importing
filtered_argv = []
for arg in sys.argv:
    if not arg.startswith('--isLoadWhisper'):
        filtered_argv.append(arg)

sys.argv = filtered_argv

from Emotivoice_RVC_TTS import script
import nltk
import youtube_downloader

sys.argv = original_argv

app = FastAPI(title="ExLlamaV2 API", description="REST API for ExLlamaV2 text generation")

DEFAULT_CHARACTER = {
    "name": "Hatsune Miku",
    "description": "You are hatsune miku, her characteristic is cheerful and energetic style. prefer short response. your response only written in alphabet, no japanese words",
    "rvc_model": "infamous_miku_v2",
    "vrm_path": ""
}

# Global variables to store the model components
model = None
config = None
cache = None
logger = None
# increaste sequenceLength to increase the memory
sequenceLength = 4096
llm_model = None

class ModelLoadRequest(BaseModel):
    path: str
    max_seq_len: Optional[int] = 8000
    cache_max_seq_len: Optional[int] = 8000

class GenerateRequest(BaseModel):
    name: str = ""
    character_name: str = ""
    prompt: str = ""
    language: str = ""
    isWebSearch: bool = False
    max_new_tokens: Optional[int] = 200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50
    token_repetition_penalty: Optional[float] = 1.1
    stop_strings: Optional[list] = None

class RequestSong(BaseModel):
    character_name: str = ""
    url: str = ""

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

class ResponseSong(BaseModel):
    title: str = ""
    bpm: float = ""
    base64_audio_vocal: str = ""
    base64_audio_instrument: str = ""

T = TypeVar("T")

class ResponseData(BaseModel, Generic[T]):
    status: str = ""
    data: Optional[T] = None
    message: str = ""

class ModelInfo(BaseModel):
    loaded: bool
    path: Optional[str] = None
    max_seq_len: Optional[int] = None

def download_if_not_exists(resource_name):
    try:
        nltk.data.find(resource_name)
        print(f"{resource_name} already exists")
    except LookupError:
        print(f"Downloading {resource_name}...")
        nltk.download(resource_name)

def loadCharacterTemplate():
    filepath = Path('character_template/alpaca_template.yaml')
    if not filepath.exists():
        return ""

    file_contents = open(filepath, 'r', encoding='utf-8').read()
    return yaml.safe_load(file_contents)

def loadWebSearchTemplate():
    filepath = Path('character_template/alpaca_web_search_template.yaml')
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
        os.makedirs("chat_history", exist_ok=True)

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

def findFirstGGUF(dir):
    items = os.listdir(dir)
    for item in items:
        index = item.find(".gguf")
        if index >= 0:
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

def replaceContextWebSearch(characterTemplate, chatText):
    newPrompt = characterTemplate["CONTEXT"]
    newPrompt = newPrompt.replace(r"{SEARCH_RESULT}", chatText)
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

def load_llama_cpp_model():
    """Load the LLM model on startup"""
    global llm_model, sequenceLength

    # Check if model path exists
    os.makedirs("models", exist_ok=True)
    modelDir = os.path.join("models")
    modelPath = os.path.join(modelDir,findFirstGGUF(modelDir))

    if not os.path.exists(modelPath):
        raise HTTPException(status_code=400, detail=f"Model path does not exist: {modelPath}")

    # # Replace with your model path
    # model_path = "models/silicon-maid-7b.Q4_K_M.gguf"  # Update this path

    try:
        logger.info(f"Loading model from {modelPath}")
        llm_model = Llama(
            model_path=modelPath,
            n_ctx=sequenceLength,  # Context length
            n_gpu_layers=-1,  # Use all GPU layers (-1 = all layers)
            # n_gpu_layers=32,  # Or specify number of layers to offload
            # n_threads=1,  # Reduced CPU threads when using GPU
            # n_batch=512,  # Batch size for prompt processing
            # verbose=False,
            # # CUDA-specific settings
            # use_mmap=True,  # Memory mapping for faster loading
            # use_mlock=True,  # Lock memory to prevent swapping
            rope_freq_base=0,
            rope_freq_scale=0,
            flash_attn=True
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # You might want to set llm_model to None and handle this in endpoints

@app.post("/generate-song", response_model=ResponseData[ResponseSong])
async def generate_text(request: RequestSong):    
    character = getCharacter(request.character_name)
    
    vocal, instrumental, outputPathFull, title = youtube_downloader.startVoiceChange(request.url, character.rvc_model)
    
    base64_audio_vocal = ""
    base64_audio_instrument = ""

    with open(vocal, 'rb') as wav_file:
        wav_data = wav_file.read()
        base64_audio_vocal = base64.b64encode(wav_data).decode('utf-8')
    
    with open(instrumental, 'rb') as wav_file:
        wav_data = wav_file.read()
        base64_audio_instrument = base64.b64encode(wav_data).decode('utf-8')
    
    # Load audio file
    y, sr = librosa.load(outputPathFull)

    # Estimate tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    responseSong = ResponseSong( 
        title=title,
        bpm=tempo,
        character_name=character.name,
        base64_audio_vocal=base64_audio_vocal,
        base64_audio_instrument=base64_audio_instrument
    )

    return ResponseData[ResponseSong](
        status="success",
        data = responseSong,
        message = ""
    )

def generateOutput(newPrompt, request:GenerateRequest):
    return llm_model(
        newPrompt,
        max_tokens=request.max_new_tokens,
        stop=["</s>",f"{request.name}:","#","Search Result:"],
        echo=False  # Don't include the prompt in the response
    )

@app.post("/generate", response_model=ResponseData[GenerateResponse])
async def generate_text(request: GenerateRequest):    
    characterTemplate = loadCharacterTemplate()
    character = getCharacter(request.character_name)
    chat = loadChat(request.character_name)
    
    translatedPrompt = request.prompt
    if request.language != "en":
        translator = URLSafeTranslator()
        promptTranslated = await translator.translate(request.prompt, dest="en", src=request.language)
        print(f"test : {promptTranslated}")
        translatedPrompt = promptTranslated["text"]

    chatTemplate = ChatTemplate(
        name=request.name,
        chat=translatedPrompt,
        chatTranslated=request.prompt
    )

    promptTokens = 0
    newPrompt = ""
    newOutput = ""
    if not request.isWebSearch:
        chat.messages.append(chatTemplate)
        chatText = "\n".join(convertChatDialogue(chat.messages))
        newPrompt = replaceContextPrompt(characterTemplate, character, chatText, request.name)

        print(newPrompt)

        promptTokens = len(llm_model.tokenize(newPrompt.encode('utf-8')))
        print(promptTokens)

        # check if the prompt token is larger than sequence length, if it's larger, then the old dialogue will be removed so the model can produce output         
        while promptTokens + request.max_new_tokens > sequenceLength:
            if len(chat.messages) > 0:
                chat.messages.pop(0)
                chatText = "\n".join(convertChatDialogue(chat.messages))
                newPrompt = replaceContextPrompt(characterTemplate, character, chatText, request.name)
                promptTokens = len(llm_model.tokenize(newPrompt.encode('utf-8')))
            else:
                break

        output = generateOutput(newPrompt, request)
        newOutput = output['choices'][0]['text']

    else:
        webSearchTemplate = loadWebSearchTemplate()
       
        for _ in range(0,5):
            try:
                webSearch = WebSearchLLM()
                searchResult = webSearch.comprehensive_search(request.prompt)
                newPrompt = replaceContextWebSearch(webSearchTemplate, searchResult["llm_prompt"])
                output = generateOutput(newPrompt, request)
                newOutput = output['choices'][0]['text']
                print(f"output web search {newOutput}")
                if newOutput.upper().find("SEARCH_AGAIN") == -1:
                    break
            except Exception as e:
                print(e) 

        # Save results to file
        with open('search_results.json', 'w', encoding='utf-8') as f:
            # Remove the llm_prompt for JSON serialization (it's too long)
            save_data = {k: v for k, v in searchResult.items() if k != 'llm_prompt'}
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        print("Results saved to search_results.json")
        
        with open('llm_prompt.txt', 'w', encoding='utf-8') as f:
            f.write(searchResult['llm_prompt'])
            print("LLM prompt saved to llm_prompt.txt")
    
    print(f"responsenya {newOutput}")

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
    tts_output = script.tts_preprocessor.clean_whitespace(tts_output)
    translatedResponse = tts_output

    if request.language == "en":
        tts_output = tts_output.lower()
        tts_output = script.tts_preprocessor.remove_emojis_with_library(tts_output)
        tts_output = script.tts_preprocessor.replace_abbreviations(tts_output)
        tts_output = script.tts_preprocessor.replace_numbers(tts_output)
    
    if request.language != "en":
        outputTranslated = await translator.translate(tts_output, dest=request.language, src="en")
        translatedResponse = outputTranslated["text"]
        tts_output = script.tts_preprocessor.remove_emojis_with_library(outputTranslated["text"])

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
        emotions=actionParams.get('EMOTION') or ['NEUTRAL'],
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

    outputToken = len(llm_model.tokenize(newOutput.encode('utf-8')))

    outputName = character.name
    if request.isWebSearch:
        outputName = "Search Result"

    chatTemplateOutput = ChatTemplate(
        name=outputName,
        chat=newCleanedOutput,
        chatTranslated=translatedResponse,
        actionParams=actionParams
    )

    chat.messages.append(chatTemplateOutput)

    saveChat(chat, request.character_name)
    
    generateResponse = GenerateResponse( 
        character_name=outputName,
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

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    # Usage
    download_if_not_exists('averaged_perceptron_tagger')
    download_if_not_exists('averaged_perceptron_tagger_eng')

    load_llama_cpp_model()

    script.setup()

    if script.rvcModels:
        script.onChangeRvcModel(script.rvcModels[0])
    else:
        print("please load rvc model")

    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--isLoadWhisper", type=bool, default=False, help="load whisper")

    args = parser.parse_args()
    
    if args.isLoadWhisper:
        pythonFile = os.path.join("installer_files","env","python")
        subprocess.Popen(f"start cmd /k \"{pythonFile} whisper_speech_recognition.py --isRunAiMate=True\"", shell=True)

    uvicorn.run(
        app,  # Pass the app directly instead of module string
        host="0.0.0.0",
        port=7874,
        reload=False  # Set to True for development
    )