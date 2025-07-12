
SET DIR=%~dp0%
%DIR%\installer_files\env\python -m pip install -r requirements.txt --upgrade
%DIR%\installer_files\env\python -m pip uninstall -y onnxruntime onnxruntime-gpu
%DIR%\installer_files\env\python -m pip install onnxruntime-gpu
%DIR%\installer_files\env\python -m pip install git+https://github.com/okazaki10/fairseq.git@main
echo done update requirements
%DIR%\ai_mate_client_installer.py