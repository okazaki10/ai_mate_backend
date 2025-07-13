
SET DIR=%~dp0%
call refreshenv
echo %CUDA_PATH%
%DIR%\installer_files\env\python -m pip install https://github.com/okazaki10/fairseq/releases/download/v.1.0-python-3.11-pytorch-2.7.0/fairseq-0.12.2-cp311-cp311-win_amd64.whl
echo done installing depedency, if you encountered error, open install_fairseq.bat again
%DIR%\add_models.bat