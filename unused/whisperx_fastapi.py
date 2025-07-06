from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import whisperx
import io
import numpy as np
from pydub import AudioSegment
from typing import Annotated
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

app = FastAPI(title="WhisperX Speech Recognition API")

# Add CORS middleware for Unity WebGL builds
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
whisper_model = None
align_model = None
metadata = None
device = "cuda"
compute_type = "float16" if device == "cuda" else "int8"

def load_models():
    """Load WhisperX models on startup"""
    global whisper_model, align_model, metadata
    
    try:
        #tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en, distil-large-v3, large-v3-turbo, turbo
        # Load WhisperX model
        whisper_model = whisperx.load_model("base", device)
        
        # Load alignment model
        # align_model, metadata = whisperx.load_align_model(device=device)
        
        print(f"Models loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading models: {e}")
        # # Fallback to smaller model if GPU memory is insufficient
        # try:
        #     whisper_model = whisperx.load_model("distil-small.en", device, compute_type=compute_type)
        #     align_model, metadata = whisperx.load_align_model(language_code="id", device=device)
        #     print(f"Fallback models loaded successfully on {device}")
        # except Exception as e2:
        #     print(f"Error loading fallback models: {e2}")

# Load models on startup
load_models()

@app.get("/")
async def root():
    return {
        "message": "WhisperX Speech Recognition API is running",
        "device": device,
        "model_loaded": whisper_model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper_model": whisper_model is not None,
        "align_model": align_model is not None,
        "device": device
    }

@app.post("/recognize")
async def recognize_speech(language: Annotated[str, Form()], audio_file: UploadFile = File(...)):
    """
    Recognize speech from uploaded audio file using WhisperX
    Supports WAV, MP3, OGG, FLAC formats
    """
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="WhisperX model not loaded")
    
    try:
        # Read the uploaded file
        audio_data = await audio_file.read()
        
        # Convert audio to WAV format if needed
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
        
        # Convert to mono and set sample rate to 16kHz (WhisperX requirement)
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        
        # Convert to numpy array for WhisperX
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / np.iinfo(np.int16).max  # Normalize to [-1, 1]
        
        print(f"language {language}")
        # Transcribe using WhisperX
        result = whisper_model.transcribe(samples, language=language)
        
        # Extract text from segments
        text = ""
        segments = []
        confidence_scores = []
        
        for segment in result["segments"]:
            text += segment["text"]
            segments.append({
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment["text"]
            })
            # WhisperX doesn't provide confidence scores in basic transcription
            confidence_scores.append(1.0)
        
        # Calculate average confidence (placeholder since WhisperX doesn't provide this)
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

        resultJson = {
            "success": True,
            "text": text.strip(),
            "segments": segments,
            "confidence": avg_confidence,
            "language": result.get("language", "en")
        }

        print(resultJson)
        
        return resultJson
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")

@app.post("/recognize_with_alignment")
async def recognize_speech_with_alignment(audio_file: UploadFile = File(...)):
    """
    Recognize speech with word-level alignment using WhisperX
    Provides timestamps for each word
    """
    if whisper_model is None or align_model is None:
        raise HTTPException(status_code=503, detail="WhisperX models not loaded")
    
    try:
        # Read the uploaded file
        audio_data = await audio_file.read()
        
        # Convert audio to WAV format if needed
        audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
        
        # Convert to mono and set sample rate to 16kHz
        audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
        
        # Convert to numpy array for WhisperX
        samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / np.iinfo(np.int16).max  # Normalize to [-1, 1]
        
        # Transcribe using WhisperX
        result = whisper_model.transcribe(samples, batch_size=16)
        
        # Perform alignment for word-level timestamps
        result = whisperx.align(result["segments"], align_model, metadata, samples, device, return_char_alignments=False)
        
        # Extract text and alignment information
        text = ""
        segments = []
        words = []
        
        for segment in result["segments"]:
            text += segment["text"]
            
            segment_data = {
                "start": segment.get("start", 0),
                "end": segment.get("end", 0),
                "text": segment["text"]
            }
            
            # Add word-level alignment if available
            if "words" in segment:
                segment_words = []
                for word in segment["words"]:
                    word_data = {
                        "word": word.get("word", ""),
                        "start": word.get("start", 0),
                        "end": word.get("end", 0),
                        "score": word.get("score", 1.0)
                    }
                    segment_words.append(word_data)
                    words.append(word_data)
                
                segment_data["words"] = segment_words
            
            segments.append(segment_data)
        
        return {
            "success": True,
            "text": text.strip(),
            "segments": segments,
            "words": words,
            "language": "en"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio with alignment: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7839)