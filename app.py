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

# Default folders per mode (used when no model_folder is specified)
_MODE_DEFAULT_FOLDER = {
    "custom": "Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "design": "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "clone_saved": "Qwen3-TTS-12Hz-1.7B-Base",
    "clone_upload": "Qwen3-TTS-12Hz-1.7B-Base",
}

# Infer output subfolder from model folder name
def _output_subfolder(folder: str) -> str:
    f = folder.lower()
    if "customvoice" in f or "custom" in f:
        return "CustomVoice"
    if "voicedesign" in f or "design" in f:
        return "VoiceDesign"
    return "Clones"

# Infer compatible mode(s) from folder name
def _folder_modes(folder: str) -> list:
    f = folder.lower()
    if "customvoice" in f or "custom" in f:
        return ["custom"]
    if "voicedesign" in f or "design" in f:
        return ["design"]
    if "base" in f:
        return ["clone_saved", "clone_upload"]
    return ["custom", "design", "clone_saved", "clone_upload"]


class GenerateRequest(BaseModel):
    mode: str                          # custom | design | clone_saved | clone_upload
    text: str
    model_folder: Optional[str] = None  # override default model for the mode
    speaker: Optional[str] = "Vivian"
    emotion: Optional[str] = "Normal tone"
    speed: Optional[float] = 1.0
    voice_description: Optional[str] = None
    voice_name: Optional[str] = None   # clone_saved
    ref_audio_b64: Optional[str] = None  # clone_upload, base64-encoded audio bytes
    ref_transcript: Optional[str] = None
    clone_instruct: Optional[str] = None  # emotion/style for cloning


def _get_model(folder: str):
    global _cached_key, _cached_model
    if _cached_key == folder and _cached_model is not None:
        return _cached_model
    _cached_model = None
    _cached_key = None
    model_path = get_smart_path(folder)
    if not model_path:
        raise FileNotFoundError(f"Model not found: models/{folder}")
    _cached_model = _load_model(model_path)
    _cached_key = folder
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
    else:
        shutil.rmtree(temp_folder, ignore_errors=True)
        raise RuntimeError(f"generate_audio produced no output at {source_file}")
    if os.path.exists(temp_folder):
        shutil.rmtree(temp_folder, ignore_errors=True)
    return os.path.join(subfolder, filename)


def _run_job(job_id: str, req: GenerateRequest):
    temp_dir = None
    try:
        if req.mode not in _MODE_DEFAULT_FOLDER:
            raise ValueError(f"Unknown mode: {req.mode}")

        folder = req.model_folder or _MODE_DEFAULT_FOLDER[req.mode]
        model = _get_model(folder)
        subfolder = _output_subfolder(folder)
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
            ref_text = ""
            if os.path.exists(txt_path):
                with open(txt_path, "r") as f:
                    ref_text = f.read().strip()
            if not ref_text:
                raise ValueError(
                    f"No transcript found for voice '{req.voice_name}'. "
                    "Re-enroll the voice and provide a transcript."
                )
            generate_audio(
                model=model, text=req.text,
                ref_audio=voice_wav, ref_text=ref_text,
                instruct=req.clone_instruct or None,
                output_path=temp_dir,
            )

        elif req.mode == "clone_upload":
            if not req.ref_audio_b64:
                raise ValueError("ref_audio_b64 is required for clone_upload mode")
            ref_text = (req.ref_transcript or "").strip()
            if not ref_text:
                raise ValueError(
                    "Transcript is required for voice cloning. "
                    "Type exactly what is said in the reference audio."
                )
            tmp_audio = f"/tmp/qwen3_ref_{job_id}.bin"
            with open(tmp_audio, "wb") as f:
                f.write(base64.b64decode(req.ref_audio_b64))
            ref_audio = convert_audio_if_needed(tmp_audio)
            if not ref_audio:
                if os.path.exists(tmp_audio):
                    os.remove(tmp_audio)
                raise ValueError("Could not process uploaded reference audio")
            try:
                generate_audio(
                    model=model, text=req.text,
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    instruct=req.clone_instruct or None,
                    output_path=temp_dir,
                )
            finally:
                if ref_audio != tmp_audio and os.path.exists(ref_audio):
                    os.remove(ref_audio)
                if os.path.exists(tmp_audio):
                    os.remove(tmp_audio)

        rel_path = _save_audio_web(temp_dir, subfolder, req.text)
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


@app.get("/models")
def list_models():
    if not os.path.exists(MODELS_DIR):
        return {"models": []}
    result = []
    for folder in sorted(os.listdir(MODELS_DIR)):
        if folder.startswith("."):
            continue
        if not os.path.isdir(os.path.join(MODELS_DIR, folder)):
            continue
        if get_smart_path(folder) is None:
            continue
        result.append({
            "folder": folder,
            "modes": _folder_modes(folder),
            "label": folder.replace("-", " "),
        })
    return {"models": result}


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
    if not os.path.realpath(full_path).startswith(os.path.realpath(BASE_OUTPUT_DIR)):
        raise HTTPException(status_code=400, detail="Invalid path")
    if not os.path.exists(full_path):
        raise HTTPException(status_code=404, detail="Audio not found")
    return FileResponse(full_path, media_type="audio/wav")


if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs("static", exist_ok=True)
    print("\n Qwen3-TTS Web Interface")
    print("-> Open http://localhost:7860 in your browser\n")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
