@echo off
setlocal

echo =^> Checking Python 3.10+...
python --version 2>nul | findstr /r "3\.[1-9][0-9]" >nul
if errorlevel 1 (
    echo Python 3.10 or newer is required.
    echo Download from: https://www.python.org/downloads/
    exit /b 1
)

echo =^> Checking ffmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ffmpeg not found. Install it with winget:
    echo   winget install ffmpeg
    echo Or download from: https://ffmpeg.org/download.html
    echo Then re-run this script.
    exit /b 1
)

echo =^> Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

echo =^> Upgrading pip...
python -m pip install --upgrade pip

echo =^> Installing PyTorch (CUDA 12.1 — remove +cu121 for CPU-only)...
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

echo =^> Installing WhisperX...
pip install whisperx

echo =^> Installing pyannote.audio...
pip install "pyannote.audio>=3.1"

echo =^> Installing Gradio and utils...
pip install "gradio>=4.0" python-dotenv

echo.
echo =^> Setup complete!
echo.
echo Next steps:
echo   1. Copy .env.example to .env and add your HuggingFace token:
echo        copy .env.example .env
echo.
echo   2. Get a free token at: https://huggingface.co/settings/tokens
echo.
echo   3. Accept model terms (one-time, click Agree on each page):
echo        https://huggingface.co/pyannote/speaker-diarization-community-1
echo.
echo   4. Start the app:
echo        run.bat

endlocal
