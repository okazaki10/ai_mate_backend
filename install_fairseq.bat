
SET DIR=%~dp0%
call refreshenv
echo %CUDA_PATH%
%DIR%\installer_files\env\python -m pip install --no-build-isolation git+https://github.com/okazaki10/fairseq.git@main
echo done installing depedency, if you encountered error, open install_depedency.bat again
%DIR%\add_models.bat