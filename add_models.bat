SET DIR=%~dp0%

git lfs pull
git lfs clone --depth 1 https://huggingface.co/WangZeJun/simbert-base-chinese %DIR%\Emotivoice_RVC_TTS\EmotiVoice\WangZeJun\simbert-base-chinese

git lfs clone --depth 1 https://www.modelscope.cn/syq163/outputs.git %DIR%\Emotivoice_RVC_TTS\EmotiVoice\outputs

git lfs clone https://huggingface.co/okazaki10/hubert_base %DIR%\Emotivoice_RVC_TTS\rvc_gui\models

git lfs clone https://huggingface.co/okazaki10/silicon_maid_7b_gguf %DIR%\models

echo done add models, if you encountered error, open add_models.bat again, if you fail at download, please delete that folder and open add_models.bat again
%DIR%\installer_files\env\python ai_mate_client_installer.py