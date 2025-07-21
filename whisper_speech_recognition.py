from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel  # Changed import
import io
import numpy as np
from pydub import AudioSegment
from typing import Annotated
import uvicorn
import argparse
import subprocess
import os

app = FastAPI(title="Faster-Whisper Speech Recognition API")

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
device = "cuda"
compute_type = "int8"

def load_models():
    """Load faster-whisper model on startup"""
    global whisper_model
    
    try:
        # Available models: tiny.en, tiny, base.en, base, small.en, small, medium.en, medium, large-v1, large-v2, large-v3, large, distil-large-v2, distil-medium.en, distil-small.en, distil-large-v3, large-v3-turbo, turbo
        whisper_model = WhisperModel(
            "large-v3-turbo", 
            device=device, 
            compute_type=compute_type,
            # Optional: specify download_root if you want to control where models are stored
            # download_root="/path/to/models"
        )
        
        print(f"Faster-whisper model loaded successfully on {device} with compute_type {compute_type}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to CPU if GPU fails
        try:
            whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
            print("Fallback model loaded successfully on CPU")
        except Exception as e2:
            print(f"Error loading fallback model: {e2}")

# Load models on startup
load_models()

@app.get("/")
async def root():
    return {
        "message": "Faster-Whisper Speech Recognition API is running",
        "device": device,
        "model_loaded": whisper_model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper_model": whisper_model is not None,
        "device": device,
        "compute_type": compute_type
    }

@app.post("/recognize")
async def recognize_speech(language: Annotated[str, Form()], audio_file: UploadFile = File(...)):
    """
    Recognize speech from uploaded audio file using faster-whisper
    Supports WAV, MP3, OGG, FLAC formats
    """
    if whisper_model is None:
        raise HTTPException(status_code=503, detail="Faster-whisper model not loaded")
    
    # Read the uploaded file
    audio_data = await audio_file.read()
    
    # Convert audio to WAV format if needed
    audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))
    
    # Convert to mono and set sample rate to 16kHz
    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
    
    # Convert to numpy array for faster-whisper
    samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
    samples = samples / np.iinfo(np.int16).max  # Normalize to [-1, 1]
    
    print(f"language {language}")
    
    # Transcribe using faster-whisper - different API than standard whisper
    # faster-whisper returns (segments, info) tuple
    segments_generator, info = whisper_model.transcribe(
        samples, 
        language=language,
        beam_size=5,  # Optional: beam search size
        best_of=5,    # Optional: number of candidates to consider
        vad_filter=True,  # Optional: voice activity detection
        # vad_parameters=dict(min_silence_duration_ms=500),  # Optional: VAD parameters
    )
    
    # Process segments (faster-whisper returns a generator)
    text = ""
    segments = []
    confidence_scores = []
    
    for segment in segments_generator:
        segment_text = segment.text
        text += segment_text
        segments.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment_text
        })
        # faster-whisper provides avg_logprob which can be used as confidence
        # Convert log probability to a more interpretable confidence score
        confidence = min(1.0, max(0.0, np.exp(segment.avg_logprob)))
        confidence_scores.append(confidence)
    
    # Calculate average confidence
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

    resultJson = {
        "success": True,
        "text": text.strip(),
        "segments": segments,
        "confidence": avg_confidence,
        "language": info.language,  # Language detected by the model
        "language_probability": info.language_probability,  # Confidence in language detection
        "duration": info.duration,  # Audio duration in seconds
        "duration_after_vad": info.duration_after_vad if hasattr(info, 'duration_after_vad') else None
    }

    print(resultJson)
    
    return resultJson

if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument("--isRunAiMate", type=bool, default=False, help="run ai mate")

    args = parser.parse_args()
    
    if args.isRunAiMate:
        aiMateFile = os.path.join("ai_mate_client","ai_mate.exe")
        subprocess.Popen(f"{aiMateFile}", shell=True)

    uvicorn.run(app, host="0.0.0.0", port=7839)