#!/bin/bash
# Starts the Gradio web UI (legacy — use run.sh for the desktop app)
set -e

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

source venv/bin/activate
python app.py
