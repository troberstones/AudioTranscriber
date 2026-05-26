"""
Generate a structured meeting summary from a transcript using a local Ollama model.
Requires Ollama to be running: https://ollama.com
"""

import ollama

_PROMPT = """\
You are an expert meeting summarizer. Given the transcript below, write a structured \
summary in Markdown with exactly these sections:

## Overview
2-3 sentences covering the purpose and outcome of the meeting.

## Key Topics
Bullet list of the main subjects discussed.

## Decisions & Action Items
Bullet list of decisions made and tasks assigned. Note who is responsible where mentioned.

## Notable Points
Any important context, concerns, or quotes worth preserving.

---

{transcript}
"""

# Truncate very long transcripts to avoid exceeding context windows
_MAX_CHARS = 80_000


def run(
    transcript: str,
    model: str = "gemma4:e4b",
    base_url: str = "http://localhost:11434",
) -> str:
    client = ollama.Client(host=base_url)
    text = transcript[:_MAX_CHARS]
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": _PROMPT.format(transcript=text)}],
    )
    return response["message"]["content"]
