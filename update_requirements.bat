
SET DIR=%~dp0%
%DIR%\installer_files\env\python -m pip install -r %DIR%\requirements.txt --upgrade
%DIR%\installer_files\env\python -m pip uninstall -y onnxruntime onnxruntime-gpu
%DIR%\installer_files\env\python -m pip install onnxruntime-gpu
echo done update requirements
%DIR%\installer_files\env\python %DIR%\ai_mate_client_installer.py
pause