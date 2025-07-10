import yt_dlp
import os
from pathlib import Path
from audio_separator.separator import Separator
from Emotivoice_RVC_TTS import script
from pydub import AudioSegment
import re

def sanitize_filename(filename):
    """
    Sanitize filename to be filesystem-safe while preserving Unicode characters
    """
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)  # Remove Windows forbidden chars
    filename = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', filename)  # Remove control characters
    filename = filename.strip('. ')  # Remove leading/trailing dots and spaces
    return filename[:200]  # Limit length to avoid filesystem limits

def download_and_separate_youtube(url, output_path="./downloads", separate_audio=True):
    """
    Download YouTube video, convert to audio, and optionally separate vocals/instruments
    
    Args:
        url (str): YouTube video URL
        output_path (str): Directory to save files
        separate_audio (bool): Whether to separate vocals and instruments
    """
    
    # Create output directories
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if separate_audio:
        Path(f"{output_path}/separated").mkdir(parents=True, exist_ok=True)
    
    # Configure yt-dlp options for high-quality audio
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'wav',
        'outtmpl': f'{output_path}/%(title)s.%(ext)s',
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown')
            
            print(f"Original Title: {title}")
            print(f"Duration: {info.get('duration', 'Unknown')} seconds")
            
            # Get the sanitized filename that yt-dlp will actually use
            sanitized_title = ydl.prepare_filename(info)
            # Remove the path and extension to get just the filename
            expected_filename = os.path.splitext(os.path.basename(sanitized_title))[0] + '.wav'
            expected_path = os.path.join(output_path, expected_filename)
            
            print(f"Expected filename: {expected_filename}")
            print("Starting download...")
            
            # Download audio
            ydl.download([url])
            
            # Find the downloaded file using multiple strategies
            audio_file = None
            
            # Strategy 1: Check the expected path
            if os.path.exists(expected_path):
                audio_file = expected_path
                print(f"Found using expected path: {expected_filename}")
            
            # Strategy 2: Look for files with similar names
            if not audio_file:
                for file in os.listdir(output_path):
                    if file.endswith('.wav'):
                        file_path = os.path.join(output_path, file)
                        # Check if this file was recently created (within last 60 seconds)
                        if os.path.getctime(file_path) > (os.path.getctime(output_path) if os.path.exists(output_path) else 0):
                            audio_file = file_path
                            break
            
            # Strategy 3: Get the most recently created .wav file
            if not audio_file:
                wav_files = []
                for file in os.listdir(output_path):
                    if file.endswith('.wav'):
                        file_path = os.path.join(output_path, file)
                        wav_files.append((file_path, os.path.getctime(file_path)))
                
                if wav_files:
                    # Sort by creation time, most recent first
                    wav_files.sort(key=lambda x: x[1], reverse=True)
                    audio_file = wav_files[0][0]
                    print(f"Using most recent .wav file: {os.path.basename(audio_file)}")
            
            if audio_file and separate_audio:
                print(f"Found audio file: {os.path.basename(audio_file)}")
                print("Starting vocal/instrument separation...")
                vocal, instrument = separate_vocals_instruments(audio_file, f"{output_path}/separated")
                return f"{output_path}/separated/{vocal}", f"{output_path}/separated/{instrument}", title
            elif audio_file:
                print(f"Audio file ready: {os.path.basename(audio_file)}")
                return audio_file, None, title
            else:
                print("Audio file not found")
                return None, None, title
                
            print(f"Process completed for: {title}")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None, None

# Alternative approach: Use a custom filename template
def download_and_separate_youtube_custom_naming(url, output_path="./downloads", separate_audio=True):
    """
    Alternative version that uses a custom naming scheme to avoid filename issues
    """
    
    # Create output directories
    Path(output_path).mkdir(parents=True, exist_ok=True)
    if separate_audio:
        Path(f"{output_path}/separated").mkdir(parents=True, exist_ok=True)
    
    try:
        with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
            # Get video info first
            info = ydl.extract_info(url, download=False)
            title = info.get('title', 'Unknown')
            video_id = info.get('id', 'unknown')
            
            # Create a safe filename using video ID + sanitized title
            safe_title = sanitize_filename(title)
            custom_filename = safe_title
            
            print(f"Original Title: {title}")
            print(f"Custom filename: {custom_filename}")
            print(f"Duration: {info.get('duration', 'Unknown')} seconds")
            print("Starting download...")
        
        # Configure yt-dlp options with custom filename
        ydl_opts = {
            'format': 'bestaudio/best',
            'extractaudio': True,
            'audioformat': 'wav',
            'outtmpl': f'{output_path}/{custom_filename}.%(ext)s',
            'noplaylist': True,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '0',
            }],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Download audio
            ydl.download([url])
            
            # The file should be exactly where we expect it
            audio_file = os.path.join(output_path, f"{custom_filename}.wav")
            
            if os.path.exists(audio_file) and separate_audio:
                print(f"Found audio file: {os.path.basename(audio_file)}")
                print("Starting vocal/instrument separation...")
                vocal, instrument = separate_vocals_instruments(audio_file, f"{output_path}/separated")
                return f"{output_path}/separated/{vocal}", f"{output_path}/separated/{instrument}", custom_filename
            elif os.path.exists(audio_file):
                print(f"Audio file ready: {os.path.basename(audio_file)}")
                return audio_file, None, title
            else:
                print("Audio file not found at expected location")
                return None, None, title
                
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return None, None, None

def separate_vocals_instruments(audio_file, output_path, model_name="UVR_MDXNET_Main.onnx"):
    """
    Separate vocals and instruments from audio file using audio-separator
    
    Args:
        audio_file (str): Path to input audio file
        output_path (str): Directory to save separated files
        model_name (str): Model to use for separation
    """
    
    try:
        # Initialize separator
        separator = Separator(
            model_file_dir='./uvr_models',  # Directory to store models
            output_dir=output_path
        )
        
        # Available models (some popular ones):
        # - "UVR-MDX-NET-Inst_HQ_3" (instruments/vocals)
        # - "UVR_MDXNET_KARA_2" (karaoke version)
        # - "Kim_Vocal_2" (vocal isolation)
        # - "kuielab_a_vocals" (vocal isolation)
        
        print(f"Using model: {model_name}")
        print("This may take several minutes depending on audio length...")
        
        # Perform separation
        separator.load_model(model_filename=model_name)
        
        # Get base filename without extension
        base_name = Path(audio_file).stem
        
        # Separate the audio
        secondary_stem_path, primary_stem_path = separator.separate(audio_file)
        
        print(f"Separation completed!")
        print(f"Vocals/Primary: {primary_stem_path}")
        print(f"Instruments/Secondary: {secondary_stem_path}")
        
        return primary_stem_path, secondary_stem_path
        
    except Exception as e:
        print(f"Error during separation: {str(e)}")
        return None, None

def batch_separate_folder(input_folder, output_folder, model_name="UVR-MDX-NET-Inst_HQ_3"):
    """
    Separate all audio files in a folder
    
    Args:
        input_folder (str): Folder containing audio files
        output_folder (str): Folder to save separated files
        model_name (str): Model to use for separation
    """
    
    supported_formats = ['.wav', '.mp3', '.flac', '.m4a', '.aac']
    audio_files = []
    
    for ext in supported_formats:
        audio_files.extend(Path(input_folder).glob(f'*{ext}'))
    
    if not audio_files:
        print("No supported audio files found!")
        return
    
    print(f"Found {len(audio_files)} audio files to process")
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\nProcessing {i}/{len(audio_files)}: {audio_file.name}")
        separate_vocals_instruments(str(audio_file), output_folder, model_name)

def download_playlist_and_separate(playlist_url, output_path="./downloads"):
    """
    Download YouTube playlist and separate each track
    
    Args:
        playlist_url (str): YouTube playlist URL
        output_path (str): Directory to save files
    """
    
    Path(output_path).mkdir(parents=True, exist_ok=True)
    Path(f"{output_path}/separated").mkdir(parents=True, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'extractaudio': True,
        'audioformat': 'wav',
        'outtmpl': f'{output_path}/%(playlist_index)s - %(title)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get playlist info
            playlist_info = ydl.extract_info(playlist_url, download=False)
            print(f"Playlist: {playlist_info.get('title', 'Unknown')}")
            print(f"Videos: {len(playlist_info.get('entries', []))}")
            
            # Download playlist
            ydl.download([playlist_url])
            
            # Separate all downloaded files
            print("\nStarting batch separation...")
            batch_separate_folder(output_path, f"{output_path}/separated")
            
    except Exception as e:
        print(f"Error occurred: {str(e)}")

def list_available_models():
    """
    List available separation models
    """
    try:
        separator = Separator()
        models = [
            "UVR-MDX-NET-Inst_HQ_3",     # High quality instrument separation
            "UVR_MDXNET_KARA_2",         # Karaoke (remove vocals)
            "Kim_Vocal_2",               # Vocal isolation
            "kuielab_a_vocals",          # Vocal isolation
            "UVR-MDX-NET-Voc_FT",       # Vocal extraction
            "UVR_MDXNET_Main",          # General purpose
        ]
        
        print("Available models:")
        for model in models:
            print(f"  - {model}")
            
        return models
        
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return []

def startVoiceChange(video_url, rvcModel):
    print("Downloading and separating single video...")
    vocal, instrument, title = download_and_separate_youtube_custom_naming(
        url=video_url,
        output_path="./music_separated",
        separate_audio=True
    )

    print(f"path vocal {vocal} {instrument}")

    audio_data = script.rvc.load_audio(vocal, 16000)

    script.rvc_click(audio_data, vocal, rvcModel)

    # Load files
    vocalWav = AudioSegment.from_wav(vocal)
    instrumentalWav = AudioSegment.from_wav(instrument)

    # Optional: match lengths (trim or loop)
    min_length = min(len(vocalWav), len(instrumentalWav))
    vocalWav = vocalWav[:min_length]
    instrumentalWav = instrumentalWav[:min_length]

    # Mix them together (overlay)
    mixed = instrumentalWav.overlay(vocalWav)

    outputPath = "rvc_music_output"
    Path(outputPath).mkdir(exist_ok=True)

    outputPathFull = f"{outputPath}/{title}_{rvcModel}.mp3"
    # Export to a new file
    mixed.export(outputPathFull, format="mp3")

    return vocal, instrument, outputPathFull, title

# Example usage
if __name__ == "__main__":
    while True:    
        print("input youtube url")
        video_url = input("youtube url: ")

        startVoiceChange(video_url, "infamous_miku_v2")
            
        # For playlist (uncomment to use):
        # playlist_url = "https://www.youtube.com/playlist?list=PLAYLIST_ID"
        # download_playlist_and_separate(playlist_url, "./playlist_separated")
        
        # For separating existing audio files (uncomment to use):
        # batch_separate_folder("./existing_music", "./separated_output")