set CMAKE_ARGS="-DGGML_CUDA=on" && installer_files\env\python -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
installer_files\env\python -m pip install -r requirements.txt --upgrade
installer_files\env\python -m pip uninstall -y onnxruntime onnxruntime-gpu
installer_files\env\python -m pip install onnxruntime-gpu
start install_chocolatey_administrator