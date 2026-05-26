"""
FastAPI backend for the Audio Transcriber web UI.
Reads HF_TOKEN from config.yaml. Streams progress via SSE.
"""

import asyncio
import json
import logging
import threading
import traceback
import uuid
from pathlib import Path
from typing import Optional

import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import diarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CONFIG_FILE = Path("config.yaml")
OUTPUTS_DIR = Path("outputs")
UPLOADS_DIR = Path("uploads")

app = FastAPI(title="Audio Transcriber")

# job_id -> {"status": "running|done|error", "progress": [...], "result": ..., "error": ...}
_jobs: dict = {}


def _load_config() -> dict:
    if not CONFIG_FILE.exists():
        return {}
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f) or {}


@app.post("/transcribe")
async def start_transcribe(
    audio: UploadFile = File(...),
    model_size: str = Form("large-v2"),
    language: Optional[str] = Form(None),
    speaker_mode: str = Form("auto"),
    num_speakers: Optional[int] = Form(None),
    min_speakers: Optional[int] = Form(None),
    max_speakers: Optional[int] = Form(None),
    include_json: bool = Form(True),
):
    config = _load_config()
    hf_token = config.get("hf_token", "").strip()
    if not hf_token:
        raise HTTPException(400, "hf_token not set in config.yaml")

    job_id = str(uuid.uuid4())
    UPLOADS_DIR.mkdir(exist_ok=True)
    suffix = Path(audio.filename or "audio.webm").suffix or ".webm"
    audio_path = UPLOADS_DIR / f"{job_id}{suffix}"
    audio_path.write_bytes(await audio.read())

    _jobs[job_id] = {"status": "running", "progress": [], "result": None, "error": None}

    def _run():
        def on_progress(frac: float, msg: str):
            _jobs[job_id]["progress"].append({"frac": frac, "msg": msg})

        try:
            kwargs: dict = {}
            if speaker_mode == "fixed" and num_speakers:
                kwargs["num_speakers"] = num_speakers
            else:
                if min_speakers:
                    kwargs["min_speakers"] = min_speakers
                if max_speakers:
                    kwargs["max_speakers"] = max_speakers

            md_path, json_path, md_text = diarize.run(
                audio_path=str(audio_path),
                hf_token=hf_token,
                model_size=model_size,
                language=language or None,
                backend="faster-whisper",
                include_json=include_json,
                output_dir=str(OUTPUTS_DIR),
                on_progress=on_progress,
                **kwargs,
            )
            _jobs[job_id]["result"] = {
                "markdown": md_text,
                "md_path": md_path,
                "json_path": json_path,
            }
            _jobs[job_id]["status"] = "done"
        except Exception as exc:
            log.error("Pipeline failed for job %s:\n%s", job_id, traceback.format_exc())
            _jobs[job_id]["error"] = str(exc)
            _jobs[job_id]["status"] = "error"
        finally:
            audio_path.unlink(missing_ok=True)

    threading.Thread(target=_run, daemon=True).start()
    return {"job_id": job_id}


@app.get("/stream/{job_id}")
async def stream_progress(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "Job not found")

    async def _generate():
        last_seen = 0
        while True:
            job = _jobs[job_id]

            while last_seen < len(job["progress"]):
                p = job["progress"][last_seen]
                yield f"event: progress\ndata: {json.dumps(p)}\n\n"
                last_seen += 1

            if job["status"] == "done":
                result = job["result"]
                payload: dict = {
                    "markdown": result["markdown"],
                    "md_url": f"/download/{job_id}/md",
                }
                if result.get("json_path"):
                    payload["json_url"] = f"/download/{job_id}/json"
                yield f"event: done\ndata: {json.dumps(payload)}\n\n"
                return

            if job["status"] == "error":
                yield f"event: pipeline_error\ndata: {json.dumps({'message': job['error']})}\n\n"
                return

            await asyncio.sleep(0.5)

    return StreamingResponse(
        _generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/download/{job_id}/{fmt}")
async def download(job_id: str, fmt: str):
    job = _jobs.get(job_id)
    if not job or job["status"] != "done":
        raise HTTPException(404)
    result = job["result"]
    if fmt == "md":
        path = result["md_path"]
        media_type = "text/markdown"
    elif fmt == "json":
        path = result.get("json_path")
        media_type = "application/json"
    else:
        raise HTTPException(400, "fmt must be md or json")
    if not path or not Path(path).exists():
        raise HTTPException(404)
    return FileResponse(path, media_type=media_type, filename=Path(path).name)


OUTPUTS_DIR.mkdir(exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


@app.on_event("startup")
async def _startup_check():
    import torch
    cuda_ok = torch.cuda.is_available()
    whisper_dev, pyannote_dev = diarize.get_device()
    log.info("=== Device check ===")
    log.info("torch.cuda.is_available() = %s", cuda_ok)
    if cuda_ok:
        log.info("CUDA device: %s", torch.cuda.get_device_name(0))
    log.info("whisper_device=%s  pyannote_device=%s", whisper_dev, pyannote_dev)
    log.info("====================")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=7860, reload=False)
