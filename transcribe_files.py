#!/usr/bin/env python3
"""
CLI for batch transcription and diarization.
Usage: transcribeFiles [options] file1.m4a file2.mp3 *.wav
"""

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── ANSI colours ──────────────────────────────────────────────────────────────
BOLD  = "\033[1m"
DIM   = "\033[2m"
GREEN = "\033[32m"
YELLOW= "\033[33m"
RED   = "\033[31m"
CYAN  = "\033[36m"
RESET = "\033[0m"

def _tty(code: str) -> str:
    return code if sys.stderr.isatty() else ""

def info(msg):  print(f"{_tty(CYAN)}  {msg}{_tty(RESET)}", file=sys.stderr)
def ok(msg):    print(f"{_tty(GREEN)}✓ {msg}{_tty(RESET)}", file=sys.stderr)
def warn(msg):  print(f"{_tty(YELLOW)}! {msg}{_tty(RESET)}", file=sys.stderr)
def err(msg):   print(f"{_tty(RED)}✗ {msg}{_tty(RESET)}", file=sys.stderr)
def header(msg):print(f"\n{_tty(BOLD)}{msg}{_tty(RESET)}", file=sys.stderr)


# ── Argument parsing ──────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="transcribeFiles",
        description="Transcribe and diarize audio/video files with WhisperX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  transcribeFiles meeting.m4a
  transcribeFiles *.m4a *.mp3
  transcribeFiles -m medium -s 3 interview.wav
  transcribeFiles --auto --min 2 --max 6 -o ~/transcripts *.m4a
        """,
    )

    p.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="audio/video files to process (shell globs are expanded automatically)",
    )

    p.add_argument(
        "-m", "--model",
        default="large-v2",
        choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
        metavar="MODEL",
        help="Whisper model size (default: large-v2)",
    )

    p.add_argument(
        "-l", "--language",
        default=None,
        metavar="LANG",
        help="language code e.g. en, fr, de (default: auto-detect)",
    )

    speaker_group = p.add_mutually_exclusive_group()
    speaker_group.add_argument(
        "-s", "--speakers",
        type=int,
        default=None,
        metavar="N",
        help="exact number of speakers (skips auto-detection)",
    )
    speaker_group.add_argument(
        "--auto",
        action="store_true",
        default=True,
        help="auto-detect speaker count (default)",
    )

    p.add_argument(
        "--min",
        type=int,
        default=None,
        metavar="N",
        help="minimum speakers hint for auto-detection",
    )
    p.add_argument(
        "--max",
        type=int,
        default=None,
        metavar="N",
        help="maximum speakers hint for auto-detection",
    )

    p.add_argument(
        "-o", "--output-dir",
        default=None,
        metavar="DIR",
        help="output directory (default: same folder as each input file)",
    )

    p.add_argument(
        "--no-json",
        action="store_true",
        help="skip JSON output, write markdown only",
    )

    p.add_argument(
        "--backend",
        default="mlx",
        choices=["mlx", "faster-whisper"],
        help="transcription backend: mlx (Apple Silicon GPU) or faster-whisper (CPU). Default: mlx",
    )

    p.add_argument(
        "--token",
        default=None,
        metavar="HF_TOKEN",
        help="HuggingFace token (default: reads HF_TOKEN from .env)",
    )

    return p


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Resolve token
    hf_token = args.token or os.getenv("HF_TOKEN", "")
    if not hf_token:
        err("No HuggingFace token found.")
        err("Set HF_TOKEN in .env or pass --token hf_...")
        sys.exit(1)

    # Resolve files — filter out anything that doesn't exist
    files = [Path(f) for f in args.files]
    missing = [f for f in files if not f.exists()]
    if missing:
        for f in missing:
            warn(f"File not found, skipping: {f}")
    files = [f for f in files if f.exists()]

    if not files:
        err("No valid files to process.")
        sys.exit(1)

    # Speaker kwargs
    if args.speakers is not None:
        speaker_kwargs = {"num_speakers": args.speakers}
    else:
        speaker_kwargs = {}
        if args.min is not None:
            speaker_kwargs["min_speakers"] = args.min
        if args.max is not None:
            speaker_kwargs["max_speakers"] = args.max

    include_json = not args.no_json

    # Import here so startup is fast for --help
    import diarize

    results = []  # (file, md_path, json_path, elapsed, error)

    total = len(files)
    for idx, audio_file in enumerate(files, 1):
        header(f"[{idx}/{total}] {audio_file.name}")

        output_dir = args.output_dir if args.output_dir else str(audio_file.parent)

        start = time.time()
        try:
            def on_progress(frac, msg):
                info(f"[{int(frac*100):3d}%] {msg}")

            md_path, json_path, _ = diarize.run(
                audio_path=str(audio_file),
                hf_token=hf_token,
                backend=args.backend,
                model_size=args.model,
                language=args.language,
                include_json=include_json,
                output_dir=output_dir,
                on_progress=on_progress,
                **speaker_kwargs,
            )

            elapsed = time.time() - start
            ok(f"Done in {elapsed:.0f}s → {md_path}")
            if json_path:
                ok(f"           → {json_path}")

            results.append((audio_file.name, md_path, json_path, elapsed, None))

        except Exception as e:
            elapsed = time.time() - start
            err(f"Failed after {elapsed:.0f}s: {e}")
            results.append((audio_file.name, None, None, elapsed, str(e)))

    # ── Summary ───────────────────────────────────────────────────────────────
    if total > 1:
        header("Summary")
        succeeded = [r for r in results if r[4] is None]
        failed    = [r for r in results if r[4] is not None]

        for name, md, _, elapsed, error in results:
            if error:
                print(f"  {_tty(RED)}FAIL{_tty(RESET)}  {name}  —  {error}", file=sys.stderr)
            else:
                print(f"  {_tty(GREEN)}OK  {_tty(RESET)}  {name}  ({elapsed:.0f}s)  →  {md}", file=sys.stderr)

        print(
            f"\n  {_tty(BOLD)}{len(succeeded)}/{total} succeeded{_tty(RESET)}",
            file=sys.stderr,
        )

    sys.exit(0 if all(r[4] is None for r in results) else 1)


if __name__ == "__main__":
    main()
