# AI MATE BACKEND
this is backend service for ai mate

# INSTALLING
## REQUIREMENTS
- NVIDIA GPU
- GPU with 8 GB of VRAM or above is recommended
- Windows 10 or above OS
## INSTALL FROM DOWNLOAD
- download from release
- open install_from_download.bat
## MANUAL INSTALL FROM GIT CLONE
- open install_from_git.bat

# RUNNING
just open start_ai_mate.bat

# FEATURES
## SINGING
- say "can you sing?"
- and then say "this is the link https://youtu.be/EXAMPLE?si=EXAMPLE"
put the youtube link from share button
  
# ADDING NEW CHARACTER
- from menu, click "character customization" button
- then change character name to something else
- change the character description
- put rvc to folder name "rvc_models", you can download rvc model from https://voice-models.com/
- download vrm model, you can download from https://booth.pm/en/items?tags%5B%5D=VRM, or download 3d model from https://sketchfab.com/feed and then convert 3d model to vrm
- after you download vrm model, you can click "load vrm" button and then load your vrm
- click "new / update character"
- click "go back

# OPTIMIZATION
if you have 12 GB VRAM or more, you can do optimization to make ai mate better

## INCREASE MEMORY
- edit api_service.py and increase `sequenceLength` number
- example from `sequenceLength = 4096` to `sequenceLength = 32000`

## BETTER SPEECH RECOGNITION MODEL
- edit whisper_speech_recognition.py and change whisper model to large-v3-turbo
- example from `whisper_model = whisper.load_model("base", device)` to `whisper_model = whisper.load_model("large-v3-turbo", device)`
