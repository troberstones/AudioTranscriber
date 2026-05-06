"""
Core WhisperX diarization pipeline.
Supports two transcription backends:
  - "faster-whisper"  CPU (ctranslate2 / int8) — works on all platforms
  - "mlx"             Apple Silicon GPU via MLX — macOS only
"""

import gc
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Callable, Optional

import torch
import whisperx
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

IS_MACOS   = sys.platform == "darwin"
IS_WINDOWS = sys.platform == "win32"

# Maps model size names → mlx-community HuggingFace repos (macOS only)
MLX_REPOS = {
    "tiny":     "mlx-community/whisper-tiny-mlx",
    "base":     "mlx-community/whisper-base-mlx",
    "small":    "mlx-community/whisper-small-mlx",
    "medium":   "mlx-community/whisper-medium-mlx",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

def importlib_check(name: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(name) is not None


MLX_AVAILABLE = IS_MACOS and importlib_check("mlx_whisper")


def get_device() -> tuple[str, str]:
    """
    Returns (whisper_device, pyannote_device).
    - faster-whisper (ctranslate2) supports CPU and CUDA, not MPS.
    - pyannote.audio supports MPS (macOS) and CUDA (Windows/Linux).
    - MLX backend handles its own device selection internally.
    """
    if torch.cuda.is_available():
        return "cuda", "cuda"
    if IS_MACOS and torch.backends.mps.is_available():
        return "cpu", "mps"   # ctranslate2 can't use MPS; pyannote can
    return "cpu", "cpu"


def _free(*objs):
    """Delete models and flush GPU memory."""
    for obj in objs:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif IS_MACOS and torch.backends.mps.is_available():
        torch.mps.empty_cache()


def format_timestamp(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def format_duration(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    parts = []
    if h > 0:
        parts.append(f"{h}h")
    if m > 0 or h > 0:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def build_markdown(segments: list[dict], audio_file: str, language: str) -> str:
    if not segments:
        return "# Meeting Transcript\n\nNo speech detected.\n"

    duration = segments[-1].get("end", 0)
    speakers = sorted({seg.get("speaker", "Unknown") for seg in segments})
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    talk_time: dict[str, float] = defaultdict(float)
    for seg in segments:
        spk = seg.get("speaker", "Unknown")
        talk_time[spk] += seg.get("end", 0) - seg.get("start", 0)

    lines = [
        "# Meeting Transcript",
        "",
        f"**Date:** {date_str}  ",
        f"**Duration:** {format_duration(duration)}  ",
        f"**Language:** {language.upper()}  ",
        f"**Speakers detected:** {len(speakers)}",
        "",
        "## Speaker Summary",
        "",
    ]
    for spk in speakers:
        pct = (talk_time[spk] / duration * 100) if duration > 0 else 0
        lines.append(f"- **{spk}**: {format_duration(talk_time[spk])} ({pct:.0f}%)")

    lines += ["", "---", "", "## Transcript", ""]

    current_speaker = None
    buffer: list[str] = []

    def flush_buffer():
        if buffer:
            lines.append(" ".join(buffer).strip())
            lines.append("")
            buffer.clear()

    for seg in segments:
        speaker = seg.get("speaker", "Unknown")
        text = seg.get("text", "").strip()
        start = seg.get("start", 0)
        if not text:
            continue
        if speaker != current_speaker:
            flush_buffer()
            current_speaker = speaker
            lines.append(f"**{speaker}** `[{format_timestamp(start)}]`")
        buffer.append(text)

    flush_buffer()
    return "\n".join(lines)


def build_json(segments: list[dict], audio_file: str, language: str) -> dict:
    duration = segments[-1].get("end", 0) if segments else 0
    speakers = sorted({seg.get("speaker", "Unknown") for seg in segments})
    return {
        "title": Path(audio_file).stem,
        "date": datetime.now().isoformat(timespec="seconds"),
        "language": language,
        "duration_seconds": round(duration, 3),
        "speakers": speakers,
        "segments": [
            {
                "speaker": seg.get("speaker", "Unknown"),
                "start": round(seg.get("start", 0), 3),
                "end": round(seg.get("end", 0), 3),
                "text": seg.get("text", "").strip(),
            }
            for seg in segments
            if seg.get("text", "").strip()
        ],
    }


def run(
    audio_path: str,
    hf_token: str,
    model_size: str = "large-v2",
    language: Optional[str] = None,
    backend: str = "mlx",
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    include_json: bool = True,
    output_dir: str = "outputs",
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> tuple[str, Optional[str], str]:
    """
    Full pipeline: transcribe → align → diarize → assign speakers → write outputs.
    on_progress(fraction: float, message: str) is called at each step.
    Returns (markdown_path, json_path_or_None, markdown_text).
    """

    def progress(frac: float, msg: str):
        if on_progress:
            on_progress(frac, msg)

    _, pyannote_device = get_device()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = Path(audio_path).stem

    # ── Step 1: Transcribe ────────────────────────────────────────────────────
    if backend == "mlx":
        import mlx_whisper
        repo = MLX_REPOS.get(model_size, MLX_REPOS["large-v2"])
        progress(0.05, f"Loading Whisper model (MLX / Apple Silicon GPU)...")
        result = mlx_whisper.transcribe(
            audio_path,
            path_or_hf_repo=repo,
            language=language,
            word_timestamps=True,
        )
        progress(0.30, "Transcription complete (MLX)")
        detected_lang = result.get("language", language or "en")
        # Load audio array for alignment (whisperx.align needs numpy array)
        audio = whisperx.load_audio(audio_path)

    else:  # faster-whisper (CPU)
        progress(0.05, "Loading Whisper model (CPU / faster-whisper)...")
        model = whisperx.load_model(
            model_size, "cpu", compute_type="int8", language=language
        )
        progress(0.10, "Transcribing audio (CPU)...")
        audio = whisperx.load_audio(audio_path)
        result = model.transcribe(audio, batch_size=8)
        _free(model)
        progress(0.30, "Transcription complete")
        detected_lang = result.get("language", language or "en")

    # ── Step 2: Align word timestamps ─────────────────────────────────────────
    progress(0.40, "Aligning word timestamps...")
    align_model, metadata = whisperx.load_align_model(
        language_code=detected_lang, device="cpu"
    )
    result = whisperx.align(
        result["segments"], align_model, metadata, audio, "cpu",
        return_char_alignments=False,
    )
    _free(align_model)

    # ── Step 3: Diarize ───────────────────────────────────────────────────────
    progress(0.60, "Running speaker diarization (Apple Silicon GPU)...")
    diarize_model = DiarizationPipeline(token=hf_token, device=pyannote_device)

    diarize_kwargs: dict = {}
    if num_speakers is not None:
        diarize_kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers is not None:
            diarize_kwargs["min_speakers"] = min_speakers
        if max_speakers is not None:
            diarize_kwargs["max_speakers"] = max_speakers

    diarize_segments = diarize_model(audio, **diarize_kwargs)
    _free(diarize_model)

    # ── Step 4: Assign speakers ───────────────────────────────────────────────
    progress(0.85, "Assigning speaker labels...")
    result = assign_word_speakers(diarize_segments, result)
    segments = result["segments"]

    # ── Step 5: Write outputs ─────────────────────────────────────────────────
    progress(0.95, "Writing outputs...")
    md_text = build_markdown(segments, audio_path, detected_lang)
    md_path = output_path / f"{stem}_{timestamp}.md"
    md_path.write_text(md_text, encoding="utf-8")

    json_path = None
    if include_json:
        json_data = build_json(segments, audio_path, detected_lang)
        json_path = output_path / f"{stem}_{timestamp}.json"
        json_path.write_text(
            json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    progress(1.0, "Done!")
    return str(md_path), str(json_path) if json_path else None, md_text
