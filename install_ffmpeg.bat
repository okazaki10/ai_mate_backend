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
    pause
    exit /b 1
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
pause