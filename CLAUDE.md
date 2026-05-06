# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
# Desktop GUI (primary)
./run.sh          # macOS
run.bat           # Windows

# CLI batch processing
./transcribeFiles meeting.m4a
python transcribe_files.py meeting.m4a   # equivalent

# Web UI (optional, port 7860)
./run_web.sh      # macOS
run_web.bat       # Windows
```

Setup (first time):
```bash
./setup.sh        # macOS — creates venv, installs deps
setup.bat         # Windows — installs CUDA PyTorch by default
cp .env.example .env   # then set HF_TOKEN=hf_...
```

There are no tests and no linter configuration.

## Architecture

`diarize.py` is the sole pipeline module. All three front-ends call `diarize.run()` and nothing else from it.

```
diarize.run()          ← single entry point for the full pipeline
    ├── gui.py         Tkinter desktop app (primary UI)
    ├── transcribe_files.py   CLI, batch-capable
    └── app.py         Gradio web UI (secondary/optional)
```

### Pipeline steps inside `diarize.run()`

1. **Transcribe** — MLX (`mlx_whisper.transcribe`) or faster-whisper (`whisperx.load_model`)
2. **Align** — `whisperx.align` (always CPU; ctranslate2 doesn't support MPS)
3. **Diarize** — `pyannote DiarizationPipeline` (uses MPS on macOS, CUDA on Windows/Linux)
4. **Assign speakers** — `assign_word_speakers`
5. **Write outputs** — `build_markdown()` + `build_json()` → `{stem}_{YYYYMMDD_HHMMSS}.md/.json`

The `on_progress(frac: float, msg: str)` callback is the only communication channel from the pipeline back to any front-end. All three front-ends implement it differently (queue, print, Gradio yield).

### Device handling

There is an intentional split: faster-whisper/whisperx alignment always runs on CPU because ctranslate2 cannot use MPS. Pyannote diarization uses MPS (macOS) or CUDA (Windows/Linux). MLX manages its own device internally. `get_device()` in `diarize.py` encodes this logic and returns `(whisper_device, pyannote_device)`.

### Backend detection

Each front-end independently checks for MLX availability at import time using `importlib.util.find_spec("mlx_whisper")` and `sys.platform == "darwin"`. This pattern is duplicated across `diarize.py`, `gui.py`, `app.py`, and `transcribe_files.py` by design — each module is independently importable.

### GUI threading model (`gui.py`)

The Tkinter app runs the pipeline in a `daemon=True` background thread. Progress and results are passed back via a `queue.Queue`. The main thread drains the queue every 200 ms via `root.after(200, self._poll_queue)`. Never call Tkinter widgets directly from the worker thread.

### Output files

Outputs are written next to the input file by default, or to the directory specified by `--output-dir` (CLI) / the output folder field (GUI). The web UI always writes to `outputs/`. Files are never overwritten — the timestamp suffix guarantees uniqueness.

## Key constraints

- **HF_TOKEN** is required at runtime for pyannote diarization. It's read from `.env` via `python-dotenv`. The `.env` file is gitignored.
- MLX backend requires macOS + Apple Silicon + `mlx-whisper` installed. faster-whisper works everywhere.
- The `transcribeFiles` file (no extension) is the installed CLI executable — it is the same code as `transcribe_files.py`.
- `*.md` and `*.json` output files are gitignored at the repo root. Add `!FILENAME.md` exceptions in `.gitignore` for any markdown docs that should be tracked.
