git lfs pull
git lfs clone --depth 1 https://huggingface.co/WangZeJun/simbert-base-chinese Emotivoice_RVC_TTS/EmotiVoice/WangZeJun/simbert-base-chinese

git lfs clone --depth 1 https://www.modelscope.cn/syq163/outputs.git Emotivoice_RVC_TTS/EmotiVoice/outputs

git lfs clone https://huggingface.co/okazaki10/hubert_base Emotivoice_RVC_TTS/rvc_gui/models

git lfs clone https://huggingface.co/okazaki10/silicon_maid_7b_gguf models

echo done add models
@REM SET DIR=%~dp0%
@REM %DIR%\install_fairseq.bat