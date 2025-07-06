git lfs pull
git lfs clone --depth 1 https://huggingface.co/WangZeJun/simbert-base-chinese Emotivoice_RVC_TTS/EmotiVoice/WangZeJun/simbert-base-chinese
rm -rf Emotivoice_RVC_TTS/EmotiVoice/outputs
git lfs clone --depth 1 https://www.modelscope.cn/syq163/outputs.git Emotivoice_RVC_TTS/EmotiVoice/outputs

git lfs pull
git lfs clone --depth 1 https://huggingface.co/kindahex/voice-conversion/blob/main/hubert_base.pt Emotivoice_RVC_TTS/rvc_gui

SET DIR=%~dp0%
start %DIR%\install_chocolatey.bat