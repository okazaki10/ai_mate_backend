
SET DIR=%~dp0%
%DIR%\installer_files\env\python -m pip install -r %DIR%\requirements.txt --upgrade
%DIR%\installer_files\env\python -m pip uninstall -y onnxruntime onnxruntime-gpu
%DIR%\installer_files\env\python -m pip install onnxruntime-gpu
%DIR%\installer_files\env\python -m pip install git+https://github.com/okazaki10/fairseq.git@main
echo done installing depedency, if you encountered error, open install_depedency.bat again
%DIR%\add_models.bat