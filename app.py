import os
import sys
import uuid
import shutil
import re
import base64
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from main import (
    MODELS, SPEAKER_MAP, EMOTION_EXAMPLES,
    BASE_OUTPUT_DIR, MODELS_DIR, VOICES_DIR, SAMPLE_RATE,
    get_smart_path, convert_audio_if_needed, get_saved_voices, make_temp_dir,
)

try:
    from mlx_audio.tts.utils import load_model as _load_model
    from mlx_audio.tts.generate import generate_audio
except ImportError:
    print("Error: mlx_audio not found. Run: source .venv/bin/activate")
    sys.exit(1)

app = FastAPI(title="Qwen3-TTS")

# In-memory job store
jobs: dict = {}

# Single model cache — load once, reload only on mode switch
_cached_key: Optional[str] = None
_cached_model = None

# One worker: TTS is memory-heavy, concurrent jobs would OOM
executor = ThreadPoolExecutor(max_workers=1)

# Map frontend mode names to MODELS keys
_MODE_TO_KEY = {"custom": "1", "design": "2", "clone_saved": "3", "clone_upload": "3"}


class GenerateRequest(BaseModel):
    mode: str                          # custom | design | clone_saved | clone_upload
    text: str
    speaker: Optional[str] = "Vivian"
    emotion: Optional[str] = "Normal tone"
    speed: Optional[float] = 1.0
    voice_description: Optional[str] = None
    voice_name: Optional[str] = None   # clone_saved
    ref_audio_b64: Optional[str] = None  # clone_upload, base64-encoded audio bytes
    ref_transcript: Optional[str] = None


def _get_model(model_key: str):
    global _cached_key, _cached_model
    if _cached_key == model_key and _cached_model is not None:
        return _cached_model
    _cached_model = None
    _cached_key = None
    model_info = MODELS[model_key]
    model_path = get_smart_path(model_info["folder"])
    if not model_path:
        raise FileNotFoundError(f"Model files not found in models/{model_info['folder']}")
    _cached_model = _load_model(model_path)
    _cached_key = model_key
    return _cached_model


def _save_audio_web(temp_folder: str, subfolder: str, text_snippet: str) -> str:
    """Save generated audio and return relative path (no auto-play)."""
    save_path = os.path.join(BASE_OUTPUT_DIR, subfolder)
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%H-%M-%S")
    clean_text = re.sub(r'[^\w\s-]', '', text_snippet)[:20].strip().replace(' ', '_') or "audio"
    filename = f"{timestamp}_{clean_text}.wav"
    final_path = os.path.join(save_path, filename)
    source_file = os.path.join(temp_folder, "audio_000.wav")
    if os.path.exists(source_file):
        shutil.move(source_file, final_path)
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder, ignore_errors=True)
    return os.path.join(subfolder, filename)


def _run_job(job_id: str, req: GenerateRequest):
    temp_dir = None
    try:
        model_key = _MODE_TO_KEY.get(req.mode)
        if not model_key:
            raise ValueError(f"Unknown mode: {req.mode}")

        model = _get_model(model_key)
        model_info = MODELS[model_key]
        temp_dir = make_temp_dir()
        os.makedirs(temp_dir, exist_ok=True)

        if req.mode == "custom":
            generate_audio(
                model=model, text=req.text,
                voice=req.speaker, instruct=req.emotion,
                speed=req.speed, output_path=temp_dir,
            )

        elif req.mode == "design":
            generate_audio(
                model=model, text=req.text,
                instruct=req.voice_description, output_path=temp_dir,
            )

        elif req.mode == "clone_saved":
            voice_wav = os.path.join(VOICES_DIR, f"{req.voice_name}.wav")
            if not os.path.exists(voice_wav):
                raise FileNotFoundError(f"Saved voice '{req.voice_name}' not found")
            txt_path = os.path.join(VOICES_DIR, f"{req.voice_name}.txt")
            ref_text = None
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    ref_text = f.read().strip()
            generate_audio(
                model=model, text=req.text,
                ref_audio=voice_wav, ref_text=ref_text, output_path=temp_dir,
            )

        elif req.mode == "clone_upload":
            if not req.ref_audio_b64:
                raise ValueError("ref_audio_b64 is required for clone_upload mode")
            tmp_audio = f"/tmp/qwen3_ref_{job_id}.bin"
            with open(tmp_audio, "wb") as f:
                f.write(base64.b64decode(req.ref_audio_b64))
            ref_audio = convert_audio_if_needed(tmp_audio)
            if not ref_audio:
                raise ValueError("Could not process uploaded reference audio")
            try:
                generate_audio(
                    model=model, text=req.text,
                    ref_audio=ref_audio,
                    ref_text=req.ref_transcript or ".",
                    output_path=temp_dir,
                )
            finally:
                if ref_audio != tmp_audio and os.path.exists(ref_audio):
                    os.remove(ref_audio)
                if os.path.exists(tmp_audio):
                    os.remove(tmp_audio)

        rel_path = _save_audio_web(temp_dir, model_info["output_subfolder"], req.text)
        jobs[job_id] = {
            "status": "done",
            "audio_url": f"/audio/{rel_path}",
            "filename": os.path.basename(rel_path),
        }

    except Exception as e:
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        jobs[job_id] = {"status": "error", "error": str(e)}


# ── Routes ──────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/generate")
def start_generate(req: GenerateRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text is required")
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "pending"}
    executor.submit(_run_job, job_id, req)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/voices")
def list_voices():
    return {"voices": get_saved_voices()}


@app.post("/voices/enroll")
async def enroll_voice(
    name: str = Form(...),
    transcript: str = Form(""),
    audio: UploadFile = File(...),
):
    safe_name = re.sub(r"[^\w\s-]", "", name).strip().replace(" ", "_")
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid voice name")

    os.makedirs(VOICES_DIR, exist_ok=True)
    tmp_path = f"/tmp/qwen3_enroll_{safe_name}.bin"
    with open(tmp_path, "wb") as f:
        f.write(await audio.read())

    wav_path = convert_audio_if_needed(tmp_path)
    if not wav_path:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(status_code=400, detail="Could not process audio (is ffmpeg installed?)")

    shutil.copy(wav_path, os.path.join(VOICES_DIR, f"{safe_name}.wav"))
    with open(os.path.join(VOICES_DIR, f"{safe_name}.txt"), "w") as f:
        f.write(transcript)

    if wav_path != tmp_path and os.path.exists(wav_path):
        os.remove(wav_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    return {"name": safe_name}


@app.delete("/voices/{name}")
def delete_voice(name: str):
    wav = os.path.join(VOICES_DIR, f"{name}.wav")
    if not os.path.exists(wav):
        raise HTTPException(status_code=404, detail="Voice not found")
    os.remove(wav)
    txt = os.path.join(VOICES_DIR, f"{name}.txt")
    if os.path.exists(txt):
        os.remove(txt)
    return {"deleted": name}


@app.get("/audio/{path:path}")
def serve_audio(path: str):
    full_path = os.path.join(BASE_OUTPUT_DIR, path)
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(full_path, media_type="audio/wav")


if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs("static", exist_ok=True)
    print("\n Qwen3-TTS Web Interface")
    print("-> Open http://localhost:7860 in your browser\n")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
