@echo off
:: Windows CLI wrapper — equivalent of the transcribeFiles shell script.
:: Usage: transcribeFiles.bat [options] file1.m4a file2.mp3 ...
::
:: To use from anywhere, add this folder to your PATH:
::   setx PATH "%PATH%;C:\path\to\wisperX"
set SCRIPT_DIR=%~dp0
"%SCRIPT_DIR%venv\Scripts\python.exe" "%SCRIPT_DIR%transcribe_files.py" %*
