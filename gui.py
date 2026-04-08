"""
WisperX — Meeting Diarizer
Lightweight Tkinter desktop UI. Runs the pipeline directly in a background
thread (no web server), identical to the CLI tool.
"""

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v2", "large-v3"]
LANGUAGES = [
    "Auto", "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru",
    "zh", "ja", "ko", "ar", "hi", "tr", "sv", "da", "fi", "no",
]


class WisperXApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("WisperX Meeting Diarizer")
        self.root.minsize(680, 700)
        self.root.resizable(True, True)

        # ── State variables ───────────────────────────────────────────────────
        self.audio_file    = tk.StringVar()
        self.hf_token      = tk.StringVar(value=os.getenv("HF_TOKEN", ""))
        self.backend       = tk.StringVar(value="mlx")
        self.model         = tk.StringVar(value="large-v2")
        self.language      = tk.StringVar(value="Auto")
        self.auto_speakers = tk.BooleanVar(value=True)
        self.num_speakers  = tk.IntVar(value=2)
        self.min_speakers  = tk.IntVar(value=2)
        self.max_speakers  = tk.IntVar(value=10)
        self.include_json  = tk.BooleanVar(value=True)
        self.output_dir    = tk.StringVar(value="")

        self.result_md_path:   str | None = None
        self.result_json_path: str | None = None
        self._queue: queue.Queue = queue.Queue()

        self._build_ui()
        self._poll_queue()

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        pad = {"padx": 10, "pady": 5}
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # ── Audio file ────────────────────────────────────────────────────────
        file_lf = ttk.LabelFrame(main, text="Audio / Video File", padding=6)
        file_lf.pack(fill=tk.X, **pad)

        self.file_entry = ttk.Entry(file_lf, textvariable=self.audio_file)
        self.file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(file_lf, text="Browse…", command=self._browse_file, width=9).pack(
            side=tk.LEFT, padx=(6, 0)
        )

        # ── Settings ──────────────────────────────────────────────────────────
        settings_lf = ttk.LabelFrame(main, text="Settings", padding=6)
        settings_lf.pack(fill=tk.X, **pad)

        # HF Token
        tok_row = ttk.Frame(settings_lf)
        tok_row.pack(fill=tk.X, pady=2)
        ttk.Label(tok_row, text="HF Token:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        self.tok_entry = ttk.Entry(tok_row, textvariable=self.hf_token, show="•")
        self.tok_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Button(tok_row, text="👁", width=3, command=self._toggle_token_vis).pack(
            side=tk.LEFT, padx=(4, 0)
        )

        # Backend
        be_row = ttk.Frame(settings_lf)
        be_row.pack(fill=tk.X, pady=2)
        ttk.Label(be_row, text="Backend:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Radiobutton(
            be_row, text="Apple Silicon GPU (MLX)",
            variable=self.backend, value="mlx"
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            be_row, text="CPU (faster-whisper)",
            variable=self.backend, value="faster-whisper"
        ).pack(side=tk.LEFT, padx=(12, 0))

        # Model + Language
        ml_row = ttk.Frame(settings_lf)
        ml_row.pack(fill=tk.X, pady=2)
        ttk.Label(ml_row, text="Model:", width=12, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Combobox(
            ml_row, textvariable=self.model, values=WHISPER_MODELS,
            state="readonly", width=12
        ).pack(side=tk.LEFT)
        ttk.Label(ml_row, text="Language:", width=10).pack(side=tk.LEFT, padx=(16, 0))
        ttk.Combobox(
            ml_row, textvariable=self.language, values=LANGUAGES,
            state="readonly", width=10
        ).pack(side=tk.LEFT)

        # ── Speakers ──────────────────────────────────────────────────────────
        spk_lf = ttk.LabelFrame(main, text="Speakers", padding=6)
        spk_lf.pack(fill=tk.X, **pad)

        mode_row = ttk.Frame(spk_lf)
        mode_row.pack(fill=tk.X)
        ttk.Radiobutton(
            mode_row, text="Auto-detect",
            variable=self.auto_speakers, value=True,
            command=self._toggle_speaker_mode
        ).pack(side=tk.LEFT)
        ttk.Radiobutton(
            mode_row, text="Fixed count",
            variable=self.auto_speakers, value=False,
            command=self._toggle_speaker_mode
        ).pack(side=tk.LEFT, padx=(16, 0))

        self.auto_spk_frame = ttk.Frame(spk_lf)
        self.auto_spk_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(self.auto_spk_frame, text="Min:").pack(side=tk.LEFT)
        ttk.Spinbox(
            self.auto_spk_frame, from_=0, to=20,
            textvariable=self.min_speakers, width=4
        ).pack(side=tk.LEFT, padx=(4, 12))
        ttk.Label(self.auto_spk_frame, text="Max:").pack(side=tk.LEFT)
        ttk.Spinbox(
            self.auto_spk_frame, from_=0, to=50,
            textvariable=self.max_speakers, width=4
        ).pack(side=tk.LEFT, padx=(4, 0))
        ttk.Label(
            self.auto_spk_frame, text="(0 = no hint)", foreground="gray"
        ).pack(side=tk.LEFT, padx=(8, 0))

        self.fixed_spk_frame = ttk.Frame(spk_lf)
        ttk.Label(self.fixed_spk_frame, text="Number of speakers:").pack(side=tk.LEFT)
        ttk.Spinbox(
            self.fixed_spk_frame, from_=1, to=50,
            textvariable=self.num_speakers, width=4
        ).pack(side=tk.LEFT, padx=(6, 0))

        # ── Output options ────────────────────────────────────────────────────
        out_lf = ttk.LabelFrame(main, text="Output", padding=6)
        out_lf.pack(fill=tk.X, **pad)

        ttk.Checkbutton(
            out_lf, text="Export JSON with timestamps",
            variable=self.include_json
        ).pack(anchor=tk.W)

        dir_row = ttk.Frame(out_lf)
        dir_row.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(dir_row, text="Output folder:", width=14, anchor=tk.W).pack(side=tk.LEFT)
        ttk.Entry(dir_row, textvariable=self.output_dir).pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        ttk.Button(dir_row, text="Browse…", command=self._browse_output, width=9).pack(
            side=tk.LEFT, padx=(6, 0)
        )
        ttk.Label(
            out_lf, text="Leave folder empty to save alongside the audio file.",
            foreground="gray"
        ).pack(anchor=tk.W, pady=(2, 0))

        # ── Run button ────────────────────────────────────────────────────────
        self.run_btn = ttk.Button(
            main, text="Transcribe & Diarize", command=self._run
        )
        self.run_btn.pack(fill=tk.X, padx=10, pady=(8, 4))

        # ── Progress ──────────────────────────────────────────────────────────
        prog_frame = ttk.Frame(main)
        prog_frame.pack(fill=tk.X, padx=10, pady=(0, 4))
        self.progress = ttk.Progressbar(prog_frame, mode="determinate", maximum=100)
        self.progress.pack(fill=tk.X)
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(prog_frame, textvariable=self.status_var, foreground="gray").pack(
            anchor=tk.W, pady=(2, 0)
        )

        # ── Transcript output ─────────────────────────────────────────────────
        out_text_lf = ttk.LabelFrame(main, text="Transcript", padding=4)
        out_text_lf.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 4))
        self.transcript = scrolledtext.ScrolledText(
            out_text_lf, height=12, state="disabled",
            wrap=tk.WORD, font=("Menlo", 11)
        )
        self.transcript.pack(fill=tk.BOTH, expand=True)

        # ── Action buttons ────────────────────────────────────────────────────
        action_frame = ttk.Frame(main)
        action_frame.pack(fill=tk.X, padx=10, pady=(0, 6))
        self.open_md_btn = ttk.Button(
            action_frame, text="Open Markdown",
            command=self._open_md, state="disabled"
        )
        self.open_md_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.open_json_btn = ttk.Button(
            action_frame, text="Open JSON",
            command=self._open_json, state="disabled"
        )
        self.open_json_btn.pack(side=tk.LEFT, padx=(0, 4))
        self.open_folder_btn = ttk.Button(
            action_frame, text="Open Output Folder",
            command=self._open_folder, state="disabled"
        )
        self.open_folder_btn.pack(side=tk.LEFT)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _toggle_token_vis(self):
        current = self.tok_entry.cget("show")
        self.tok_entry.config(show="" if current == "•" else "•")

    def _toggle_speaker_mode(self):
        if self.auto_speakers.get():
            self.fixed_spk_frame.pack_forget()
            self.auto_spk_frame.pack(fill=tk.X, pady=(4, 0))
        else:
            self.auto_spk_frame.pack_forget()
            self.fixed_spk_frame.pack(fill=tk.X, pady=(4, 0))

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Audio / Video File",
            filetypes=[
                ("Audio/Video", "*.m4a *.mp3 *.wav *.mp4 *.mov *.aac *.ogg *.flac *.wma"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.audio_file.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self.output_dir.set(path)

    def _set_transcript(self, text: str):
        self.transcript.config(state="normal")
        self.transcript.delete("1.0", tk.END)
        self.transcript.insert("1.0", text)
        self.transcript.config(state="disabled")

    def _open_md(self):
        if self.result_md_path:
            subprocess.Popen(["open", self.result_md_path])

    def _open_json(self):
        if self.result_json_path:
            subprocess.Popen(["open", self.result_json_path])

    def _open_folder(self):
        if self.result_md_path:
            subprocess.Popen(["open", str(Path(self.result_md_path).parent)])

    # ── Pipeline ──────────────────────────────────────────────────────────────

    def _run(self):
        audio = self.audio_file.get().strip()
        if not audio:
            messagebox.showerror("Missing File", "Please select an audio file.")
            return
        if not Path(audio).exists():
            messagebox.showerror("File Not Found", f"Cannot find:\n{audio}")
            return

        token = self.hf_token.get().strip()
        if not token:
            messagebox.showerror(
                "Missing Token",
                "A HuggingFace token is required.\n"
                "Get one free at huggingface.co/settings/tokens\n\n"
                "Paste it in the HF Token field or add HF_TOKEN to your .env file."
            )
            return

        # Reset UI state
        self.run_btn.config(state="disabled")
        self.open_md_btn.config(state="disabled")
        self.open_json_btn.config(state="disabled")
        self.open_folder_btn.config(state="disabled")
        self.progress["value"] = 0
        self.status_var.set("Starting…")
        self.result_md_path = None
        self.result_json_path = None
        self._set_transcript("")

        output_dir = self.output_dir.get().strip() or str(Path(audio).parent)

        speaker_kwargs: dict = {}
        if not self.auto_speakers.get():
            speaker_kwargs["num_speakers"] = self.num_speakers.get()
        else:
            mn = self.min_speakers.get()
            mx = self.max_speakers.get()
            if mn > 0:
                speaker_kwargs["min_speakers"] = mn
            if mx > 0:
                speaker_kwargs["max_speakers"] = mx

        q = self._queue

        def on_progress(frac: float, msg: str):
            q.put(("progress", frac, msg))

        def worker():
            try:
                import diarize
                md_path, json_path, md_text = diarize.run(
                    audio_path=audio,
                    hf_token=token,
                    backend=self.backend.get(),
                    model_size=self.model.get(),
                    language=None if self.language.get() == "Auto" else self.language.get(),
                    include_json=self.include_json.get(),
                    output_dir=output_dir,
                    on_progress=on_progress,
                    **speaker_kwargs,
                )
                q.put(("done", md_path, json_path, md_text))
            except Exception as exc:
                q.put(("error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self):
        """Drain the queue on the main thread every 200 ms."""
        try:
            while True:
                item = self._queue.get_nowait()
                kind = item[0]

                if kind == "progress":
                    _, frac, msg = item
                    self.progress["value"] = int(frac * 100)
                    self.status_var.set(msg)

                elif kind == "done":
                    _, md_path, json_path, md_text = item
                    self.result_md_path = md_path
                    self.result_json_path = json_path
                    self.progress["value"] = 100
                    self.status_var.set("Complete!")
                    self._set_transcript(md_text)
                    self.run_btn.config(state="normal")
                    self.open_md_btn.config(state="normal")
                    self.open_folder_btn.config(state="normal")
                    if json_path:
                        self.open_json_btn.config(state="normal")

                elif kind == "error":
                    _, msg = item
                    self.progress["value"] = 0
                    self.status_var.set("Failed.")
                    self.run_btn.config(state="normal")
                    messagebox.showerror("Pipeline Error", msg)

        except queue.Empty:
            pass

        self.root.after(200, self._poll_queue)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    root = tk.Tk()
    app = WisperXApp(root)
    root.mainloop()
