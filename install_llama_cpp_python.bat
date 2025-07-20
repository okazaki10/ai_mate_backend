
SET DIR=%~dp0%
%DIR%\installer_files\env\python -m pip install ninja && %DIR%\installer_files\env\python -m pip install py-cpuinfo==9.0.0
%DIR%\installer_files\env\python -m pip install torch==2.7.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
@REM set CMAKE_ARGS="-DGGML_CUDA=on" && %DIR%\installer_files\env\python -m pip install llama-cpp-python==0.3.9 --upgrade --force-reinstall --no-cache-dir
%DIR%\installer_files\env\python -m pip install https://github.com/okazaki10/llama-cpp-python/releases/download/0.3.9/llama_cpp_python-0.3.9-cp311-cp311-win_amd64.whl
echo done installing pytorch and llama cpp python, if you encountered error, open install_llama_cpp_python.bat again
%DIR%\install_depedency.bat