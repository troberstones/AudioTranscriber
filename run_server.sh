#!/usr/bin/env bash
set -e

if [ ! -f config.yaml ]; then
  echo "ERROR: config.yaml not found. Copy config.example.yaml and set your hf_token."
  exit 1
fi

if [ -d venv ]; then
  source venv/bin/activate
fi

uvicorn server:app --host 0.0.0.0 --port 7860
