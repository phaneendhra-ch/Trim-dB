@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if "%~1"=="" (
    echo Usage: %~nx0 "D:\Raw_Videos\Mohan_Videos\2026\09Sept\02Sept2026"
    echo.
    echo Pass the folder that contains the .mp4 files.
    exit /b 1
)

if not exist "%~dp0env_torch_audio_gpu_1\Scripts\activate.bat" (
    echo Error: venv not found:
    echo   %~dp0env_torch_audio_gpu_1\Scripts\activate.bat
    exit /b 1
)

if not exist "%~dp0process_videos_batch_gpu_full.py" (
    echo Error: process_videos_batch_gpu_full.py not found in:
    echo   %~dp0
    exit /b 1
)

call "%~dp0env_torch_audio_gpu_1\Scripts\activate.bat"
if errorlevel 1 (
    echo Error: failed to activate env_torch_audio_gpu_1
    exit /b 1
)

python -u "%~dp0process_videos_batch_gpu_full.py" "%~1"
exit /b %ERRORLEVEL%
