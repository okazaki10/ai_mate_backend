
SET DIR=%~dp0%
%DIR%\installer_files\env\python -m pip install ninja && %DIR%\installer_files\env\python -m pip install py-cpuinfo==9.0.0
%DIR%\installer_files\env\python -m pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
@REM set CMAKE_ARGS="-DGGML_CUDA=on" && %DIR%\installer_files\env\python -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
echo done installing pytorch and llama cpp python
%DIR%\install_depedency.bat