"""
WisperX — Meeting Diarization Web UI
Run with: ./run.sh  (or: source venv/bin/activate && python app.py)
"""

import os
import time
import threading
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

import diarize

load_dotenv()

DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN", "")
OUTPUTS_DIR = "outputs"

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
LANGUAGES = [
    "Auto",
    "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru",
    "zh", "ja", "ko", "ar", "hi", "tr", "sv", "da", "fi", "no",
]

# Pipeline steps and their approximate completion fractions
STEPS = [
    (0.05, "Loading Whisper model..."),
    (0.30, "Transcribing audio..."),
    (0.40, "Aligning word timestamps..."),
    (0.60, "Running speaker diarization..."),
    (0.85, "Assigning speaker labels..."),
    (0.95, "Writing outputs..."),
    (1.00, "Done!"),
]


def run_pipeline(
    audio_file,
    hf_token,
    backend,
    model_size,
    language_choice,
    auto_speakers,
    num_speakers,
    min_speakers,
    max_speakers,
    include_json,
    progress=gr.Progress(),
):
    # ── Validation ────────────────────────────────────────────────────────────
    if not hf_token or not hf_token.strip():
        raise gr.Error(
            "HuggingFace token is required. "
            "Get one free at huggingface.co/settings/tokens"
        )
    if audio_file is None:
        raise gr.Error("Please upload an audio file or record one with the microphone.")

    # Disable button, clear outputs
    yield (
        gr.update(value="Processing...", interactive=False),
        gr.update(value=""),
        gr.update(value=None, visible=False),
        gr.update(value=None, visible=False),
        gr.update(value="Starting...", visible=True),
    )

    language = None if language_choice == "Auto" else language_choice
    speakers_kwargs: dict = {}
    if not auto_speakers:
        speakers_kwargs["num_speakers"] = int(num_speakers)
    else:
        mn, mx = int(min_speakers), int(max_speakers)
        if mn > 0:
            speakers_kwargs["min_speakers"] = mn
        if mx > 0:
            speakers_kwargs["max_speakers"] = mx

    # ── Run pipeline in background thread ─────────────────────────────────────
    progress_updates: list[tuple[float, str]] = []
    result_holder: list = [None]
    error_holder: list[str] = [None]

    def on_progress(frac: float, msg: str):
        progress_updates.append((frac, msg))

    def worker():
        try:
            result_holder[0] = diarize.run(
                audio_path=audio_file,
                hf_token=hf_token.strip(),
                backend="mlx" if backend == "Apple Silicon GPU (MLX)" else "faster-whisper",
                model_size=model_size,
                language=language,
                include_json=include_json,
                output_dir=OUTPUTS_DIR,
                on_progress=on_progress,
                **speakers_kwargs,
            )
        except Exception as e:
            error_holder[0] = str(e)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    # ── Poll for progress and update the UI ───────────────────────────────────
    last_seen = 0
    while thread.is_alive():
        time.sleep(1.0)
        if len(progress_updates) > last_seen:
            last_seen = len(progress_updates)
            frac, msg = progress_updates[-1]
            progress(frac, desc=msg)
            yield (
                gr.update(value="Processing...", interactive=False),
                gr.update(value=""),
                gr.update(value=None, visible=False),
                gr.update(value=None, visible=False),
                gr.update(value=msg, visible=True),
            )

    thread.join()

    if error_holder[0]:
        raise gr.Error(f"Pipeline failed: {error_holder[0]}")

    md_path, json_path, md_text = result_holder[0]
    progress(1.0, desc="Done!")

    yield (
        gr.update(value="Transcribe & Diarize", interactive=True),
        gr.update(value=md_text),
        gr.update(value=md_path, visible=True),
        gr.update(value=json_path, visible=(json_path is not None)),
        gr.update(value="Complete!", visible=True),
    )


# ─── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="WisperX Meeting Diarizer") as app:
    gr.Markdown("# WisperX Meeting Diarizer")
    gr.Markdown(
        "Upload or record a meeting and get a speaker-attributed transcript."
    )

    with gr.Row():
        # ── Left column ───────────────────────────────────────────────────────
        with gr.Column(scale=1):
            audio_input = gr.Audio(
                label="Audio / Video File — upload or record",
                type="filepath",
                sources=["microphone", "upload"],
            )

            hf_token_input = gr.Textbox(
                label="HuggingFace Token",
                value=DEFAULT_HF_TOKEN,
                type="password",
                placeholder="hf_...",
                info=(
                    "Required for speaker diarization. "
                    "Free at huggingface.co/settings/tokens — "
                    "accept model terms at pyannote/speaker-diarization-community-1"
                ),
            )

            backend_radio = gr.Radio(
                label="Transcription Backend",
                choices=["Apple Silicon GPU (MLX)", "CPU (faster-whisper)"],
                value="Apple Silicon GPU (MLX)",
                info=(
                    "MLX uses the M-series GPU/Neural Engine. "
                    "faster-whisper uses CPU with int8 quantization."
                ),
            )

            model_dropdown = gr.Dropdown(
                label="Whisper Model",
                choices=WHISPER_MODELS,
                value="large-v2",
                info="larger = more accurate, slower",
            )

            language_dropdown = gr.Dropdown(
                label="Language",
                choices=LANGUAGES,
                value="Auto",
                info="Set manually to skip auto-detection (slightly faster)",
            )

        # ── Right column ──────────────────────────────────────────────────────
        with gr.Column(scale=1):
            auto_speakers_toggle = gr.Checkbox(
                label="Auto-detect speaker count",
                value=True,
            )

            with gr.Group() as auto_group:
                gr.Markdown("**Hints for auto-detection** (set 0 to skip):")
                with gr.Row():
                    min_speakers_input = gr.Number(
                        label="Min speakers", value=2,
                        minimum=0, maximum=20, step=1, precision=0,
                    )
                    max_speakers_input = gr.Number(
                        label="Max speakers", value=10,
                        minimum=0, maximum=50, step=1, precision=0,
                    )

            with gr.Group(visible=False) as fixed_group:
                num_speakers_input = gr.Number(
                    label="Number of speakers", value=2,
                    minimum=1, maximum=50, step=1, precision=0,
                )

            include_json_toggle = gr.Checkbox(
                label="Also export JSON with timestamps",
                value=True,
            )

            run_btn = gr.Button(
                "Transcribe & Diarize",
                variant="primary",
                size="lg",
            )

            status_box = gr.Textbox(
                label="Status",
                interactive=False,
                visible=False,
            )

    # ── Output section ────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column():
            transcript_box = gr.Textbox(
                label="Transcript",
                lines=25,
                interactive=False,
            )

        with gr.Column(scale=0, min_width=180):
            gr.Markdown("### Download")
            md_download = gr.File(label="Markdown", visible=False)
            json_download = gr.File(label="JSON", visible=False)

    gr.Markdown(
        "---\n"
        "Outputs are saved to the `outputs/` folder.  \n"
        "**MLX** runs Whisper on the M-series GPU. "
        "**Diarization** always uses the M-series GPU via MPS."
    )

    # ── Interactions ──────────────────────────────────────────────────────────

    def toggle_speaker_mode(auto: bool):
        return gr.update(visible=auto), gr.update(visible=not auto)

    auto_speakers_toggle.change(
        toggle_speaker_mode,
        inputs=[auto_speakers_toggle],
        outputs=[auto_group, fixed_group],
    )

    run_btn.click(
        run_pipeline,
        inputs=[
            audio_input,
            hf_token_input,
            backend_radio,
            model_dropdown,
            language_dropdown,
            auto_speakers_toggle,
            num_speakers_input,
            min_speakers_input,
            max_speakers_input,
            include_json_toggle,
        ],
        outputs=[run_btn, transcript_box, md_download, json_download, status_box],
    )


if __name__ == "__main__":
    Path(OUTPUTS_DIR).mkdir(exist_ok=True)
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        max_threads=2,
    )
