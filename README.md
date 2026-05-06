# AudioTranscriber

Transcribe and diarize audio recordings into speaker-labeled markdown using [WhisperX](https://github.com/m-bain/whisperX) and [pyannote.audio](https://github.com/pyannote/pyannote-audio).

Supports a desktop GUI, a CLI for batch processing, and an optional Gradio web UI. Runs on macOS (Apple Silicon MLX or CPU) and Windows/Linux (CUDA or CPU).

## Features

- Speaker diarization — identifies and labels each speaker in the transcript
- Auto-detects number of speakers, or you can specify an exact count
- Two transcription backends: **MLX** (Apple Silicon GPU) and **faster-whisper** (CPU/CUDA)
- Outputs clean markdown and JSON for each recording
- Desktop GUI (Tkinter) and CLI for batch processing

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/)
- A free [HuggingFace token](https://huggingface.co/settings/tokens) with access to the pyannote models (see setup below)

## Setup

### macOS

```bash
./setup.sh
```

### Windows

```bat
setup.bat
```

Then configure your HuggingFace token:

```bash
cp .env.example .env   # macOS/Linux
copy .env.example .env  # Windows
```

Edit `.env` and paste your token:

```
HF_TOKEN=hf_your_token_here
```

**One-time model approval** — visit each link and click Agree:
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0

## Usage

### Desktop GUI

```bash
./run.sh        # macOS
run.bat         # Windows
```

Select an audio file, configure options, and click **Transcribe**.

### CLI (batch)

```bash
./transcribeFiles meeting.m4a
./transcribeFiles *.m4a *.mp3
./transcribeFiles -m medium -s 3 interview.wav
./transcribeFiles --auto --min 2 --max 6 -o ~/transcripts *.m4a
```

**Key options:**

| Flag | Description |
|------|-------------|
| `-m`, `--model` | Whisper model size: `tiny` `base` `small` `medium` `large-v2` `large-v3` (default: `large-v2`) |
| `-s`, `--speakers` | Exact speaker count (skips auto-detection) |
| `--min` / `--max` | Speaker count hints for auto-detection |
| `-l`, `--language` | Language code e.g. `en`, `fr` (default: auto-detect) |
| `-o`, `--output-dir` | Output folder (default: same folder as input file) |
| `--backend` | `mlx` or `faster-whisper` |
| `--no-json` | Skip JSON output, write markdown only |

### Web UI (optional)

```bash
./run_web.sh    # macOS
run_web.bat     # Windows
```

## Output

Each audio file produces:

- `filename_YYYYMMDD_HHMMSS.md` — speaker-labeled transcript in markdown
- `filename_YYYYMMDD_HHMMSS.json` — full structured output with timestamps

## License

MIT
