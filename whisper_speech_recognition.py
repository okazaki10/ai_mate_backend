import argparse
import os
import numpy as np
import speech_recognition as sr
from faster_whisper import WhisperModel
import asyncio
import websockets
import json
import threading
import subprocess

from datetime import datetime, timedelta
from queue import Queue
from time import sleep
from sys import platform
from pydantic import BaseModel
from pathlib import Path

class UserConfig(BaseModel):
    language: str = "en"
    
class WebSocketSpeechRecognizer:
    def __init__(self):
        self.clients = set()
        self.transcription = ['']
        self.phrase_time = None
        self.data_queue = Queue()
        self.phrase_bytes = bytes()
        self.setup_speech_recognition()
        self.setup_whisper_model()
        self.isConnected = False
        
    def setup_speech_recognition(self):
        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = 1000
        self.recorder.dynamic_energy_threshold = False
        
        # Setup microphone
        if 'linux' in platform:
            # For Linux, you might need to specify microphone
            self.source = sr.Microphone(sample_rate=16000)
        else:
            self.source = sr.Microphone(sample_rate=16000)
            
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)
    
    def setup_whisper_model(self):
        self.audio_model = WhisperModel(
            "large-v3-turbo", 
            device="cuda", 
            compute_type="int8"
        )

    def loadConfig(self) -> UserConfig:
        try:
            os.makedirs("user_config", exist_ok=True)

            filepath = Path(f"user_config/user_config.json")
            if not filepath.exists():
                with open(filepath, "w") as json_file:
                    json.dump(UserConfig().model_dump(), json_file, indent=4)
            
            with open(filepath, "r") as json_file:
                data = json.load(json_file)
                return UserConfig(**data)
        except Exception as e:
            return UserConfig()
        return UserConfig()
    
    def record_callback(self, _, audio: sr.AudioData) -> None:
        """Threaded callback function to receive audio data when recordings finish."""
        if not self.isConnected:
            return
        data = audio.get_raw_data()
        self.data_queue.put(data)

    async def register_client(self, websocket):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        print(f"Client connected: {websocket.remote_address}")
        sleep(1)
        self.isConnected = True
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            print(f"Client disconnected: {websocket.remote_address}")
            self.isConnected = False

    async def broadcast_transcription(self, text, is_final=False):
        """Send transcription to all connected clients"""
        if self.clients:
            message = {
                "type": "transcription",
                "text": text,
                "is_final": is_final,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to all clients
            disconnected_clients = set()
            for client in self.clients:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
            
            # Remove disconnected clients
            self.clients -= disconnected_clients

    def process_audio(self):
        """Process audio data in a separate thread"""
        record_timeout = 2
        phrase_timeout = 3
        idleTime = datetime.utcnow()
        isTranscribing = False
        phrase_complete = False
        
        # Start listening in background
        self.recorder.listen_in_background(
            self.source, 
            self.record_callback, 
            phrase_time_limit=record_timeout
        )
        
        print("Speech recognition started. Model loaded.\n")
        
        while True:
            try:
                if not self.data_queue.empty() and self.isConnected: 
                    isTranscribing = True
                    idleTime = datetime.utcnow()
                    
                    # Combine audio data from queue
                    audio_data = b''.join(self.data_queue.queue)
                    self.data_queue.queue.clear()
                    
                    # Add new audio data to accumulated data
                    self.phrase_bytes += audio_data
                    
                    # Convert to numpy array for Whisper
                    audio_np = np.frombuffer(self.phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    config = self.loadConfig()
                    print(f"language {config.language}")

                    # Transcribe audio
                    result, info = self.audio_model.transcribe(
                        audio_np, 
                        language=config.language,
                        beam_size=5,
                        best_of=5,
                        vad_filter=True
                    )

                    text = ""
                    for t in result:
                        text += t.text
                    
                    # Update transcription
                    if phrase_complete:
                        if self.transcription[-1]:  # Only add if previous line has content
                            self.transcription.append(text)
                        else:
                            self.transcription[-1] = text
                        is_final = True
                        phrase_complete = False
                    else:
                        self.transcription[-1] = text
                        is_final = False
                    
                    # Send to WebSocket clients
                    if text.strip():  # Only send if there's actual text
                        asyncio.run_coroutine_threadsafe(
                            self.broadcast_transcription(text.strip(), False),
                            self.loop
                        )

                    # Clear console and print updated transcription
                    os.system('cls' if os.name=='nt' else 'clear')
                    for line in self.transcription:
                        if line.strip():
                            print(line)
                    print('', end='', flush=True)
                else:   
                    if isTranscribing and self.isConnected and ((datetime.utcnow() - idleTime) > timedelta(seconds=phrase_timeout)):
                        print("done")
                        self.phrase_bytes = bytes()
                        phrase_complete = True
                        if self.transcription[-1]:
                            asyncio.run_coroutine_threadsafe(
                                self.broadcast_transcription(self.transcription[-1], True),
                                self.loop
                            )
                        isTranscribing = False
                    sleep(0.25)
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in audio processing: {e}")
                sleep(1)

    async def start_server(self, host='localhost', port=8765):
        """Start the WebSocket server"""
        self.loop = asyncio.get_event_loop()
        
        # Start audio processing in a separate thread
        audio_thread = threading.Thread(target=self.process_audio, daemon=True)
        audio_thread.start()
        
        # Start WebSocket server
        print(f"Starting WebSocket server on ws://{host}:{port}")
        async with websockets.serve(self.register_client, host, port):
            await asyncio.Future()  # Run forever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default='localhost', help="WebSocket server host")
    parser.add_argument("--port", default=8765, type=int, help="WebSocket server port")
    parser.add_argument("--isRunAiMate", type=bool, default=False, help="run ai mate")

    args = parser.parse_args()
    
    # Create and start the WebSocket speech recognizer
    recognizer = WebSocketSpeechRecognizer()

    if args.isRunAiMate:
        aiMateFile = os.path.join("ai_mate_client","ai_mate.exe")
        subprocess.Popen(f"{aiMateFile}", shell=True)
    
    try:
        asyncio.run(recognizer.start_server(args.host, args.port))
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()