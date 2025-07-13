
SET DIR=%~dp0%
call refreshenv
%DIR%\installer_files\env\python -m pip install git+https://github.com/okazaki10/fairseq.git@main
echo done installing depedency, if you encountered error, open install_depedency.bat again
%DIR%\add_models.bat