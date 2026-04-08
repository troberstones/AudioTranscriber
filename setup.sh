#!/bin/bash
set -e

echo "==> Installing Python 3.11..."
brew install python@3.11

PYTHON=/opt/homebrew/bin/python3.11

echo "==> Creating virtual environment..."
$PYTHON -m venv venv
source venv/bin/activate

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing PyTorch (Apple Silicon)..."
pip install torch torchaudio

echo "==> Installing WhisperX and dependencies..."
pip install whisperx

echo "==> Installing pyannote.audio..."
pip install "pyannote.audio>=3.1"

echo "==> Installing Gradio and utils..."
pip install "gradio>=4.0" python-dotenv

echo ""
echo "==> Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Copy .env.example to .env and add your HuggingFace token:"
echo "       cp .env.example .env"
echo ""
echo "  2. Get a free HF token at: https://huggingface.co/settings/tokens"
echo ""
echo "  3. Accept model terms (required, one-time):"
echo "       https://huggingface.co/pyannote/speaker-diarization-3.1"
echo "       https://huggingface.co/pyannote/segmentation-3.0"
echo ""
echo "  4. Start the app:"
echo "       ./run.sh"
