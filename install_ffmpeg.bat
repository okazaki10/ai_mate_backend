@echo off
echo Installing FFmpeg via Chocolatey...
echo ==================================

SET DIR=%~dp0%

:: Check if chocolatey is installed
choco --version >nul 2>&1
if errorlevel 1 (
    echo Chocolatey not found. please run install_chocolatey.bat
    echo.
)

echo Chocolatey found. Installing FFmpeg...

:: Install FFmpeg using Chocolatey
choco install ffmpeg -y

if errorlevel 1 (
    echo FFmpeg installation failed.
)

:: Refresh environment to update PATH
call refreshenv

echo.
echo Verifying installation...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo FFmpeg not found in PATH. Please restart your command prompt.
) else (
    echo ✓ FFmpeg installed successfully!
)

ffprobe -version >nul 2>&1
if errorlevel 1 (
    echo FFprobe not found in PATH. Please restart your command prompt.
) else (
    echo ✓ FFprobe installed successfully!
)

echo.
echo Installation complete!
echo You can now use 'ffmpeg' and 'ffprobe' from any command prompt.

choco install git -y

if errorlevel 1 (
    echo git installation failed.
)

:: Refresh environment to update PATH
call refreshenv

echo.
echo Verifying installation...
git -v >nul 2>&1
if errorlevel 1 (
    echo git not found in PATH. Please restart your command prompt.
) else (
    echo ✓ git installed successfully!
)

echo.
echo Installation complete!
echo You can now use git

choco install git-lfs.install -y

if errorlevel 1 (
    echo git installation failed.
)

:: Refresh environment to update PATH
call refreshenv

echo.
echo Verifying installation...
git -v >nul 2>&1
if errorlevel 1 (
    echo git not found in PATH. Please restart your command prompt.
) else (
    echo ✓ git installed successfully!
)

echo.
echo Installation complete!
echo You can now use git lfs

start install_llama_cpp_python.bat
pause