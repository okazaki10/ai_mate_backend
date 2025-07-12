import os, subprocess
import sys
import requests
import zipfile
import shutil
from pathlib import Path

# Environment
script_dir = Path(__file__).resolve().parent
conda_env_path = os.path.join(script_dir, "installer_files", "env")
python_path = os.path.join(script_dir, "installer_files", "env", "python")

def run_command_as_admin(command):
    powershell_command = f'Start-Process -FilePath "cmd" -ArgumentList "/c {command}" -Verb RunAs -Wait'
    
    subprocess.run([
        'powershell', 
        '-Command', 
        powershell_command
    ])

def is_linux():
    return sys.platform.startswith("linux")


def is_windows():
    return sys.platform.startswith("win")

def download_file(url, filename):
    """Download file with progress indication"""
    print(f"Downloading {filename}...")
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end='', flush=True)
        
        print(f"\nDownload completed: {filename}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}")
        return False

def extract_zip(zip_path, extract_to):
    """Extract zip file to specified directory"""
    print(f"Extracting {zip_path} to {extract_to}...")
    
    try:
        # Create destination directory if it doesn't exist
        Path(extract_to).mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        print(f"Extraction completed to: {extract_to}")
        return True
        
    except zipfile.BadZipFile:
        print("Error: Invalid zip file")
        return False
    except Exception as e:
        print(f"Extraction failed: {e}")
        return False


def move_bin_files(extract_path, final_path):
    """Move files from bin directory to final destination"""
    bin_path = os.path.join(extract_path, "cudnn-windows-x86_64-8.9.7.29_cuda12-archive", "bin")
    
    if not os.path.exists(bin_path):
        print(f"Warning: bin directory not found at {bin_path}")
        return False
    
    print(f"Moving files from {bin_path} to {final_path}...")
    
    try:
        # Create final destination directory if it doesn't exist
        Path(final_path).mkdir(parents=True, exist_ok=True)
        
        # Get list of files in bin directory
        bin_files = [f for f in os.listdir(bin_path) if os.path.isfile(os.path.join(bin_path, f))]
        
        if not bin_files:
            print("No files found in bin directory")
            return False
        
        # Move each file
        for filename in bin_files:
            src_file = os.path.join(bin_path, filename)
            dst_file = os.path.join(final_path, filename)
            
            # If destination file exists, remove it first
            if os.path.exists(dst_file):
                os.remove(dst_file)
            
            shutil.move(src_file, dst_file)
            print(f"  Moved: {filename}")
        
        print(f"✓ Successfully moved {len(bin_files)} files to {final_path}")
        return True
        
    except Exception as e:
        print(f"Error moving files: {e}")
        return False

def downloadFiles(url, filename, final_path, isDeleteTemp = True):
    final_path = os.path.join(script_dir, final_path)
    
    filepath = Path(final_path)
    if filepath.exists():
        print(f"path exist {filepath}")
        if isDeleteTemp:
            shutil.rmtree(filepath)
        else:
            return
    
    print("Download")
    print("=" * 40)
    print(f"URL: {url}")
    print(f"Final destination: {final_path}")
    print()
    
    # Check if requests module is available
    try:
        import requests
    except ImportError:
        print("Error: 'requests' module not found. Please install it with:")
        print("pip install requests")
        sys.exit(1)
    
    # Download the file
    if download_file(url, filename):
        print()
        
        # Extract the file to temporary directory
        parent = filepath.parent
        if extract_zip(filename, parent):
            print()
            tempPath = Path(filename)
            tempPath.unlink()
            
            # # Move bin files to final destination
            # if move_bin_files(temp_extract_path, final_path):
            #     print()
            #     print("✓ Download, extraction, and file moving completed successfully!")
                
            #     # Clean up temporary files and directories
            #     try:
            #         os.remove(filename)
            #         shutil.rmtree(temp_extract_path)
            #         print(f"✓ Cleaned up temporary files and directories")
            #     except Exception as e:
            #         print(f"Note: Could not clean up temporary files: {e}")
            # else:
            #     print("✗ File moving failed")
            #     sys.exit(1)
        else:
            print("✗ Extraction failed")
            sys.exit(1)
    else:
        print("✗ Download failed")
        sys.exit(1)

def check_env():
    # If we have access to conda, we are probably in an environment
    conda_exist = run_cmd("conda", environment=True, capture_output=True).returncode == 0
    if not conda_exist:
        print("Conda is not installed. Exiting...")
        sys.exit(1)

    # Ensure this is a new environment and not the base environment
    if os.environ.get("CONDA_DEFAULT_ENV", "") == "base":
        print("Create an environment for this project and activate it. Exiting...")
        sys.exit(1)

def run_cmd(cmd, assert_success=False, environment=False, capture_output=False, env=None):
    # # Use the conda environment
    # if environment:
    #     if is_windows():
    #         conda_bat_path = os.path.join(script_dir, "installer_files", "conda", "condabin", "conda.bat")
    #         cmd = f'"{conda_bat_path}" activate "{conda_env_path}" >nul && {cmd}'
    #     else:
    #         conda_sh_path = os.path.join(script_dir, "installer_files", "conda", "etc", "profile.d", "conda.sh")
    #         cmd = f'. "{conda_sh_path}" && conda activate "{conda_env_path}" && {cmd}'
    # # Set executable to None for Windows, bash for everything else
    # executable = None if is_windows() else 'bash'
    # cmd = f"{conda_env_path}/{cmd}"

    # Run shell commands
    result = subprocess.run(cmd, shell=True, capture_output=capture_output, env=env)

    # Assert the command ran successfully
    if assert_success and result.returncode != 0:
        print(f"Command '{cmd}' failed with exit status code '{str(result.returncode)}'.\n\nExiting now.\nTry running the start/update script again.")
        sys.exit(1)

    return result

if __name__ == "__main__":
    # download ai mate client
    print("downloading ai mate client")
    downloadFiles("https://github.com/okazaki10/ai_mate_client/releases/download/v.1.0/ai_mate_client.zip", "ai_mate_client_temp.zip", "ai_mate_client")

    # download rvc model
    print("downloading rvc model")
    downloadFiles("https://huggingface.co/javinfamous/infamous_miku_v2/resolve/main/infamous_miku_v2.zip", "infamous_miku_v2_temp.zip", "rvc_models\infamous_miku_v2", False)
    
    print("installation is done, you can open start_ai_mate.bat")