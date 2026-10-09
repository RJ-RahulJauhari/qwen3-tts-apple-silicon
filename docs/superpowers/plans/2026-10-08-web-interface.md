# Qwen3-TTS Web Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a local FastAPI web server + single-page browser UI that exposes all three TTS modes (Custom Voice, Voice Design, Voice Cloning) with inline audio playback, sidebar layout, and non-blocking generation via job polling.

**Architecture:** FastAPI serves `static/index.html` at `GET /` and handles generation as background jobs in a `ThreadPoolExecutor`. The frontend polls `GET /status/{job_id}` every 500 ms. All TTS logic is imported from `main.py` — no duplication.

**Tech Stack:** FastAPI, Uvicorn, python-multipart, plain HTML/CSS/JS (no build step), mlx_audio (existing)

## Global Constraints

- Run from project root with venv active: `python app.py`
- Apple Silicon only (mlx_audio)
- Only 1.7B models (keys "1", "2", "3" in MODELS dict) — 0.6B commented out
- All paths relative to project root (same cwd as CLI)
- No auth, localhost only (127.0.0.1:7860)
- No auto-play of audio server-side (browser handles playback)
- Reuse `get_smart_path`, `convert_audio_if_needed`, `get_saved_voices`, `make_temp_dir` from `main.py`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `requirements.txt` | Modify | Add fastapi, uvicorn, python-multipart |
| `app.py` | Create | FastAPI server: endpoints, job system, model cache |
| `static/index.html` | Create | Full SPA: HTML + CSS + JS, all inline |

---

### Task 1: Add dependencies and verify server starts

**Files:**
- Modify: `requirements.txt`
- Create: `app.py` (skeleton only)
- Create: `static/` directory

**Interfaces:**
- Produces: `GET /health` → `{"status": "ok"}`

- [ ] **Step 1: Add new packages to requirements.txt**

Append these three lines to `requirements.txt`:
```
fastapi==0.115.12
uvicorn==0.34.3
python-multipart==0.0.20
```

- [ ] **Step 2: Install new packages**

```bash
pip install fastapi==0.115.12 uvicorn==0.34.3 python-multipart==0.0.20
```

Expected: installs cleanly with no conflicts.

- [ ] **Step 3: Create static/ directory**

```bash
mkdir -p static
```

- [ ] **Step 4: Create app.py skeleton**

Create `app.py`:

```python
import os
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_ui():
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static", "index.html")
    with open(html_path, "r") as f:
        return HTMLResponse(content=f.read())


if __name__ == "__main__":
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)
    os.makedirs("static", exist_ok=True)
    print("\n Qwen3-TTS Web Interface")
    print("-> Open http://localhost:7860 in your browser\n")
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")
```

- [ ] **Step 5: Create a placeholder index.html so the server starts**

Create `static/index.html`:

```html
<!DOCTYPE html>
<html><body><h1>Qwen3-TTS — coming soon</h1></body></html>
```

- [ ] **Step 6: Verify server starts**

```bash
python app.py &
sleep 3
curl -s http://localhost:7860/health
```

Expected output: `{"status":"ok"}`

```bash
kill %1
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt app.py static/index.html
git commit -m "feat: add FastAPI server skeleton for web UI"
```

---

### Task 2: Full backend — generation, voices, audio serving

**Files:**
- Modify: `app.py`

**Interfaces:**
- Consumes: `get_smart_path`, `convert_audio_if_needed`, `get_saved_voices`, `make_temp_dir` from `main.py`; `_load_model`, `generate_audio` from mlx_audio
- Produces:
  - `POST /generate` → `{"job_id": "<uuid>"}`
  - `GET /status/{job_id}` → `{"status": "pending"|"done"|"error", "audio_url"?: str, "filename"?: str, "error"?: str}`
  - `GET /voices` → `{"voices": ["name1", "name2"]}`
  - `POST /voices/enroll` (multipart: `name`, `transcript`, `audio` file) → `{"name": str}`
  - `DELETE /voices/{name}` → `{"deleted": str}`
  - `GET /audio/{path}` → WAV file stream

- [ ] **Step 1: Replace app.py with the full backend**

Overwrite `app.py` with:

```python
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
        if os.path.exists(temp_dir if 'temp_dir' in dir() else ''):
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
```

- [ ] **Step 2: Smoke-test the API endpoints with curl**

```bash
python app.py &
sleep 3

# Health
curl -s http://localhost:7860/health
# Expected: {"status":"ok"}

# Voices list (empty or populated from voices/ dir)
curl -s http://localhost:7860/voices
# Expected: {"voices":[...]}

# Unknown job
curl -s http://localhost:7860/status/nonexistent
# Expected: {"detail":"Job not found"}

# Start a generation job (will fail if model not loaded yet — that's fine, status will show error)
curl -s -X POST http://localhost:7860/generate \
  -H "Content-Type: application/json" \
  -d '{"mode":"custom","text":"Hello world","speaker":"Vivian","emotion":"Normal tone","speed":1.0}'
# Expected: {"job_id":"<uuid>"}

kill %1
```

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add FastAPI backend with generate/voices/audio endpoints"
```

---

### Task 3: Frontend — full static/index.html (HTML + CSS + JS)

**Files:**
- Modify: `static/index.html` (replace placeholder with full SPA)

**Interfaces:**
- Consumes: all endpoints from Task 2
- Produces: working browser UI at `http://localhost:7860`

- [ ] **Step 1: Write the complete static/index.html**

Replace `static/index.html` with:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Qwen3-TTS</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg: #0d0d0f;
  --sidebar-bg: #111113;
  --panel-bg: #18181b;
  --border: #2a2a2e;
  --accent: #7c6df0;
  --accent-hover: #6a5ce0;
  --text: #e4e4e7;
  --text-muted: #71717a;
  --text-dim: #52525b;
  --success: #22c55e;
  --error: #ef4444;
  --input-bg: #1c1c1f;
  --radius: 8px;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  display: flex;
  overflow: hidden;
}

/* ── Sidebar ── */
.sidebar {
  width: 220px;
  min-width: 220px;
  background: var(--sidebar-bg);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  padding: 0;
}

.sidebar-header {
  padding: 20px 16px 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
}

.sidebar-logo {
  font-size: 22px;
}

.sidebar-title {
  font-size: 15px;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.3px;
}

.sidebar-subtitle {
  font-size: 11px;
  color: var(--text-muted);
  margin-top: 1px;
}

.nav-list {
  list-style: none;
  padding: 10px 8px;
  flex: 1;
}

.nav-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 12px;
  border-radius: var(--radius);
  cursor: pointer;
  color: var(--text-muted);
  font-size: 14px;
  font-weight: 500;
  transition: background 0.15s, color 0.15s;
  user-select: none;
}

.nav-item:hover { background: var(--panel-bg); color: var(--text); }

.nav-item.active {
  background: rgba(124, 109, 240, 0.15);
  color: var(--accent);
}

.nav-icon { font-size: 16px; width: 20px; text-align: center; }

.sidebar-footer {
  padding: 12px 16px;
  border-top: 1px solid var(--border);
  font-size: 11px;
  color: var(--text-dim);
}

/* ── Main panel ── */
.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.panel-wrap {
  flex: 1;
  overflow-y: auto;
  padding: 28px 32px 16px;
}

.panel { display: none; }
.panel.active { display: block; }

.panel-title {
  font-size: 20px;
  font-weight: 700;
  margin-bottom: 24px;
  color: var(--text);
}

/* ── Form elements ── */
.form-group { margin-bottom: 18px; }

label {
  display: block;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-muted);
  margin-bottom: 6px;
}

input[type="text"],
input[type="file"],
select,
textarea {
  width: 100%;
  background: var(--input-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
  font-size: 14px;
  padding: 9px 12px;
  outline: none;
  transition: border-color 0.15s;
  font-family: inherit;
}

input[type="text"]:focus,
select:focus,
textarea:focus { border-color: var(--accent); }

select { cursor: pointer; }
textarea { resize: vertical; min-height: 90px; line-height: 1.5; }

.radio-row {
  display: flex;
  gap: 16px;
}

.radio-row label {
  display: flex;
  align-items: center;
  gap: 6px;
  cursor: pointer;
  color: var(--text);
  font-size: 14px;
  font-weight: 400;
  margin-bottom: 0;
}

input[type="radio"] { accent-color: var(--accent); cursor: pointer; }

/* ── Buttons ── */
.btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 9px 18px;
  border-radius: var(--radius);
  border: none;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.15s, opacity 0.15s;
}

.btn-primary {
  background: var(--accent);
  color: #fff;
}
.btn-primary:hover { background: var(--accent-hover); }
.btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }

.btn-ghost {
  background: transparent;
  color: var(--text-muted);
  border: 1px solid var(--border);
}
.btn-ghost:hover { color: var(--text); border-color: var(--text-muted); }

.btn-danger {
  background: transparent;
  color: var(--error);
  border: 1px solid transparent;
  padding: 4px 8px;
  font-size: 12px;
}
.btn-danger:hover { background: rgba(239,68,68,0.1); }

/* ── Clone sub-tabs ── */
.clone-tabs {
  display: flex;
  gap: 4px;
  margin-bottom: 20px;
  border-bottom: 1px solid var(--border);
  padding-bottom: 0;
}

.clone-tab {
  background: none;
  border: none;
  color: var(--text-muted);
  font-size: 13px;
  font-weight: 500;
  padding: 8px 14px;
  cursor: pointer;
  border-bottom: 2px solid transparent;
  margin-bottom: -1px;
  transition: color 0.15s, border-color 0.15s;
}

.clone-tab:hover { color: var(--text); }
.clone-tab.active { color: var(--accent); border-bottom-color: var(--accent); }

/* ── Saved voices list ── */
.voices-list {
  margin-bottom: 16px;
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.voice-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: var(--input-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 10px 12px;
  cursor: pointer;
  transition: border-color 0.15s;
}

.voice-item:hover { border-color: var(--text-muted); }
.voice-item.selected { border-color: var(--accent); background: rgba(124,109,240,0.08); }

.voice-item-name { font-size: 14px; font-weight: 500; }

.empty-state {
  text-align: center;
  color: var(--text-dim);
  font-size: 13px;
  padding: 24px 0;
  border: 1px dashed var(--border);
  border-radius: var(--radius);
  margin-bottom: 16px;
}

/* ── Enroll form ── */
.enroll-form {
  background: var(--panel-bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px;
  margin-bottom: 16px;
}

.enroll-form .form-group:last-of-type { margin-bottom: 12px; }

.enroll-actions { display: flex; gap: 8px; }

/* ── Drop zone ── */
.drop-zone {
  border: 2px dashed var(--border);
  border-radius: var(--radius);
  padding: 24px;
  text-align: center;
  color: var(--text-muted);
  font-size: 13px;
  cursor: pointer;
  transition: border-color 0.15s, background 0.15s;
  margin-bottom: 14px;
}

.drop-zone:hover, .drop-zone.drag-over {
  border-color: var(--accent);
  background: rgba(124,109,240,0.06);
}

.drop-zone input[type="file"] { display: none; }
.drop-file-name { margin-top: 6px; font-size: 12px; color: var(--success); }

/* ── Output area ── */
.output-area {
  border-top: 1px solid var(--border);
  padding: 16px 32px;
  min-height: 74px;
  display: flex;
  align-items: center;
  gap: 16px;
  background: var(--sidebar-bg);
}

.output-placeholder {
  color: var(--text-dim);
  font-size: 13px;
}

.status-spinner {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  color: var(--text-muted);
  font-size: 13px;
}

.spinner {
  width: 18px; height: 18px;
  border: 2px solid var(--border);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}

@keyframes spin { to { transform: rotate(360deg); } }

.audio-wrap {
  display: flex;
  align-items: center;
  gap: 12px;
  width: 100%;
}

audio {
  flex: 1;
  height: 36px;
  accent-color: var(--accent);
}

.audio-filename {
  font-size: 11px;
  color: var(--text-dim);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 160px;
}

.error-msg {
  color: var(--error);
  font-size: 13px;
  background: rgba(239,68,68,0.08);
  border: 1px solid rgba(239,68,68,0.2);
  border-radius: var(--radius);
  padding: 10px 14px;
  width: 100%;
}

.hint-text {
  font-size: 11px;
  color: var(--text-dim);
  margin-top: 4px;
}
</style>
</head>
<body>

<!-- Sidebar -->
<nav class="sidebar">
  <div class="sidebar-header">
    <span class="sidebar-logo">🎙</span>
    <div>
      <div class="sidebar-title">Qwen3-TTS</div>
      <div class="sidebar-subtitle">1.7B · Apple Silicon</div>
    </div>
  </div>
  <ul class="nav-list">
    <li class="nav-item active" data-mode="custom" onclick="switchMode('custom')">
      <span class="nav-icon">🎭</span> Custom Voice
    </li>
    <li class="nav-item" data-mode="design" onclick="switchMode('design')">
      <span class="nav-icon">✏️</span> Voice Design
    </li>
    <li class="nav-item" data-mode="clone" onclick="switchMode('clone')">
      <span class="nav-icon">👤</span> Voice Cloning
    </li>
  </ul>
  <div class="sidebar-footer">localhost:7860</div>
</nav>

<!-- Main -->
<div class="main">
  <div class="panel-wrap">

    <!-- Custom Voice panel -->
    <section id="panel-custom" class="panel active">
      <div class="panel-title">Custom Voice</div>

      <div class="form-group">
        <label>Speaker</label>
        <select id="speaker-select">
          <optgroup label="English">
            <option>Ryan</option><option>Aiden</option><option>Ethan</option>
            <option>Chelsie</option><option>Serena</option><option selected>Vivian</option>
          </optgroup>
          <optgroup label="Chinese">
            <option>Uncle_Fu</option><option>Dylan</option><option>Eric</option>
          </optgroup>
          <optgroup label="Japanese">
            <option>Ono_Anna</option>
          </optgroup>
          <optgroup label="Korean">
            <option>Sohee</option>
          </optgroup>
        </select>
      </div>

      <div class="form-group">
        <label>Emotion / Style</label>
        <input type="text" id="emotion-input" placeholder="e.g. Excited and happy, speaking very fast">
        <div class="hint-text">Leave blank for normal tone</div>
      </div>

      <div class="form-group">
        <label>Speed</label>
        <div class="radio-row">
          <label><input type="radio" name="speed" value="0.8"> Slow (0.8×)</label>
          <label><input type="radio" name="speed" value="1.0" checked> Normal (1.0×)</label>
          <label><input type="radio" name="speed" value="1.3"> Fast (1.3×)</label>
        </div>
      </div>

      <div class="form-group">
        <label>Text to synthesize</label>
        <textarea id="custom-text" rows="5" placeholder="Enter the text you want to convert to speech..."></textarea>
      </div>

      <button class="btn btn-primary" id="btn-custom" onclick="submitGenerate('custom')">Generate</button>
    </section>

    <!-- Voice Design panel -->
    <section id="panel-design" class="panel">
      <div class="panel-title">Voice Design</div>

      <div class="form-group">
        <label>Describe the voice</label>
        <textarea id="design-desc" rows="3" placeholder="e.g. A warm, friendly female voice with a slight British accent, speaking at a moderate, measured pace"></textarea>
      </div>

      <div class="form-group">
        <label>Text to synthesize</label>
        <textarea id="design-text" rows="5" placeholder="Enter the text you want to convert to speech..."></textarea>
      </div>

      <button class="btn btn-primary" id="btn-design" onclick="submitGenerate('design')">Generate</button>
    </section>

    <!-- Voice Cloning panel -->
    <section id="panel-clone" class="panel">
      <div class="panel-title">Voice Cloning</div>

      <div class="clone-tabs">
        <button class="clone-tab active" id="ctab-saved" onclick="switchCloneTab('saved')">Saved Voices</button>
        <button class="clone-tab" id="ctab-upload" onclick="switchCloneTab('upload')">Quick Upload</button>
      </div>

      <!-- Saved voices sub-panel -->
      <div id="cpanel-saved">
        <div class="voices-list" id="voices-list"></div>

        <button class="btn btn-ghost" style="margin-bottom:16px" onclick="toggleEnrollForm()">+ Enroll New Voice</button>

        <div class="enroll-form" id="enroll-form" style="display:none">
          <div class="form-group">
            <label>Voice name</label>
            <input type="text" id="enroll-name" placeholder="e.g. Boss, Mom, Client">
          </div>
          <div class="form-group">
            <label>Reference audio file</label>
            <input type="file" id="enroll-audio" accept="audio/*" style="padding:6px">
          </div>
          <div class="form-group">
            <label>Transcript <span style="color:var(--text-dim)">(type exactly what the audio says)</span></label>
            <textarea id="enroll-transcript" rows="2" placeholder="The quick brown fox..."></textarea>
          </div>
          <div class="enroll-actions">
            <button class="btn btn-primary" onclick="enrollVoice()">Save Voice</button>
            <button class="btn btn-ghost" onclick="toggleEnrollForm()">Cancel</button>
          </div>
        </div>
      </div>

      <!-- Quick upload sub-panel -->
      <div id="cpanel-upload" style="display:none">
        <div class="drop-zone" id="drop-zone" onclick="document.getElementById('upload-audio').click()"
             ondragover="onDragOver(event)" ondragleave="onDragLeave(event)" ondrop="onDrop(event)">
          <div>Drag &amp; drop audio file here</div>
          <div style="color:var(--text-dim);margin-top:4px;font-size:12px">or click to browse (WAV, MP3, M4A…)</div>
          <input type="file" id="upload-audio" accept="audio/*" onchange="onFileSelected(this)">
          <div class="drop-file-name" id="drop-file-name"></div>
        </div>
        <div class="form-group">
          <label>Transcript <span style="color:var(--text-dim)">(optional but improves quality)</span></label>
          <textarea id="upload-transcript" rows="2" placeholder="What the reference audio says..."></textarea>
        </div>
      </div>

      <div class="form-group" style="margin-top:8px">
        <label>Text to synthesize</label>
        <textarea id="clone-text" rows="5" placeholder="Enter the text you want to convert to speech..."></textarea>
      </div>

      <button class="btn btn-primary" id="btn-clone" onclick="submitGenerate('clone')">Generate</button>
    </section>

  </div><!-- /panel-wrap -->

  <!-- Output area -->
  <div class="output-area" id="output-area">
    <div class="output-placeholder" id="output-placeholder">Output will appear here after generation</div>
  </div>
</div><!-- /main -->

<script>
// ── State ──────────────────────────────────────────────────────────────────
let currentMode = 'custom';
let currentCloneTab = 'saved';
let selectedVoice = null;
let uploadedFile = null;
let pollHandle = null;

// ── Mode switching ─────────────────────────────────────────────────────────
function switchMode(mode) {
  currentMode = mode;
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('panel-' + mode).classList.add('active');
  document.querySelector(`.nav-item[data-mode="${mode}"]`).classList.add('active');
  if (mode === 'clone') fetchVoices();
}

// ── Clone sub-tabs ─────────────────────────────────────────────────────────
function switchCloneTab(tab) {
  currentCloneTab = tab;
  document.getElementById('cpanel-saved').style.display = tab === 'saved' ? '' : 'none';
  document.getElementById('cpanel-upload').style.display = tab === 'upload' ? '' : 'none';
  document.getElementById('ctab-saved').classList.toggle('active', tab === 'saved');
  document.getElementById('ctab-upload').classList.toggle('active', tab === 'upload');
}

// ── Voices ─────────────────────────────────────────────────────────────────
async function fetchVoices() {
  const listEl = document.getElementById('voices-list');
  try {
    const res = await fetch('/voices');
    const data = await res.json();
    renderVoicesList(data.voices);
  } catch (e) {
    listEl.innerHTML = '<div class="empty-state">Failed to load voices</div>';
  }
}

function renderVoicesList(voices) {
  const listEl = document.getElementById('voices-list');
  if (!voices || voices.length === 0) {
    listEl.innerHTML = '<div class="empty-state">No saved voices yet — enroll one below</div>';
    return;
  }
  listEl.innerHTML = voices.map(v => `
    <div class="voice-item ${selectedVoice === v ? 'selected' : ''}" onclick="selectVoice('${v}')">
      <span class="voice-item-name">👤 ${v}</span>
      <button class="btn btn-danger" onclick="deleteVoice(event,'${v}')">✕ Remove</button>
    </div>
  `).join('');
}

function selectVoice(name) {
  selectedVoice = name;
  document.querySelectorAll('.voice-item').forEach(el => el.classList.remove('selected'));
  const items = document.querySelectorAll('.voice-item');
  items.forEach(el => {
    if (el.querySelector('.voice-item-name').textContent.includes(name)) {
      el.classList.add('selected');
    }
  });
}

async function deleteVoice(event, name) {
  event.stopPropagation();
  if (!confirm(`Remove voice "${name}"?`)) return;
  await fetch(`/voices/${encodeURIComponent(name)}`, { method: 'DELETE' });
  if (selectedVoice === name) selectedVoice = null;
  fetchVoices();
}

function toggleEnrollForm() {
  const form = document.getElementById('enroll-form');
  form.style.display = form.style.display === 'none' ? '' : 'none';
}

async function enrollVoice() {
  const name = document.getElementById('enroll-name').value.trim();
  const audioFile = document.getElementById('enroll-audio').files[0];
  const transcript = document.getElementById('enroll-transcript').value.trim();
  if (!name) { alert('Voice name is required'); return; }
  if (!audioFile) { alert('Reference audio file is required'); return; }

  const fd = new FormData();
  fd.append('name', name);
  fd.append('transcript', transcript);
  fd.append('audio', audioFile);

  const btn = event.target;
  btn.disabled = true;
  btn.textContent = 'Saving…';

  try {
    const res = await fetch('/voices/enroll', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Enroll failed');
    }
    document.getElementById('enroll-name').value = '';
    document.getElementById('enroll-audio').value = '';
    document.getElementById('enroll-transcript').value = '';
    toggleEnrollForm();
    fetchVoices();
  } catch (e) {
    alert('Error: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Save Voice';
  }
}

// ── Drop zone ──────────────────────────────────────────────────────────────
function onDragOver(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.add('drag-over');
}
function onDragLeave(e) {
  document.getElementById('drop-zone').classList.remove('drag-over');
}
function onDrop(e) {
  e.preventDefault();
  document.getElementById('drop-zone').classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setUploadFile(file);
}
function onFileSelected(input) {
  if (input.files[0]) setUploadFile(input.files[0]);
}
function setUploadFile(file) {
  uploadedFile = file;
  document.getElementById('drop-file-name').textContent = '✓ ' + file.name;
}

// ── Generation ─────────────────────────────────────────────────────────────
async function submitGenerate(mode) {
  const btnId = 'btn-' + mode;
  const btn = document.getElementById(btnId);

  let body = {};

  if (mode === 'custom') {
    const text = document.getElementById('custom-text').value.trim();
    if (!text) { alert('Please enter text to synthesize'); return; }
    const speed = parseFloat(document.querySelector('input[name="speed"]:checked').value);
    body = {
      mode: 'custom',
      text,
      speaker: document.getElementById('speaker-select').value,
      emotion: document.getElementById('emotion-input').value.trim() || 'Normal tone',
      speed,
    };

  } else if (mode === 'design') {
    const text = document.getElementById('design-text').value.trim();
    const desc = document.getElementById('design-desc').value.trim();
    if (!text) { alert('Please enter text to synthesize'); return; }
    if (!desc) { alert('Please describe the voice'); return; }
    body = { mode: 'design', text, voice_description: desc };

  } else if (mode === 'clone') {
    const text = document.getElementById('clone-text').value.trim();
    if (!text) { alert('Please enter text to synthesize'); return; }

    if (currentCloneTab === 'saved') {
      if (!selectedVoice) { alert('Please select a saved voice'); return; }
      body = { mode: 'clone_saved', text, voice_name: selectedVoice };

    } else {
      if (!uploadedFile) { alert('Please upload a reference audio file'); return; }
      const b64 = await fileToBase64(uploadedFile);
      body = {
        mode: 'clone_upload',
        text,
        ref_audio_b64: b64,
        ref_transcript: document.getElementById('upload-transcript').value.trim() || '.',
      };
    }
  }

  btn.disabled = true;
  btn.textContent = 'Generating…';
  showStatus('generating');

  try {
    const res = await fetch('/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Request failed');
    }
    const { job_id } = await res.json();
    startPolling(job_id, btn);
  } catch (e) {
    showError(e.message);
    btn.disabled = false;
    btn.textContent = 'Generate';
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // result is "data:<mime>;base64,<data>" — strip the prefix
      resolve(reader.result.split(',')[1]);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function startPolling(jobId, btn) {
  if (pollHandle) clearInterval(pollHandle);
  pollHandle = setInterval(async () => {
    try {
      const res = await fetch('/status/' + jobId);
      const job = await res.json();
      if (job.status === 'done') {
        clearInterval(pollHandle);
        btn.disabled = false;
        btn.textContent = 'Generate';
        showAudioPlayer(job.audio_url, job.filename);
      } else if (job.status === 'error') {
        clearInterval(pollHandle);
        btn.disabled = false;
        btn.textContent = 'Generate';
        showError(job.error || 'Generation failed');
      }
      // pending → keep polling
    } catch (e) {
      clearInterval(pollHandle);
      btn.disabled = false;
      btn.textContent = 'Generate';
      showError('Lost connection to server');
    }
  }, 500);
}

// ── Output area rendering ──────────────────────────────────────────────────
function showStatus(state) {
  const area = document.getElementById('output-area');
  area.innerHTML = `
    <div class="status-spinner">
      <div class="spinner"></div>
      <span>Generating audio… this may take a few seconds</span>
    </div>`;
}

function showAudioPlayer(audioUrl, filename) {
  const area = document.getElementById('output-area');
  area.innerHTML = `
    <div class="audio-wrap">
      <audio controls src="${audioUrl}" style="flex:1"></audio>
      <span class="audio-filename" title="${filename}">${filename}</span>
      <a href="${audioUrl}" download="${filename}" class="btn btn-ghost" style="white-space:nowrap">↓ Download</a>
    </div>`;
}

function showError(msg) {
  const area = document.getElementById('output-area');
  area.innerHTML = `<div class="error-msg">⚠ ${msg}</div>`;
}

// ── Init ───────────────────────────────────────────────────────────────────
// Reset output area to placeholder on page load
document.getElementById('output-area').innerHTML =
  '<div class="output-placeholder">Output will appear here after generation</div>';
</script>
</body>
</html>
```

- [ ] **Step 2: Start the server and open the browser**

```bash
python app.py
```

Open `http://localhost:7860` in your browser. Verify:
- Sidebar shows three items: Custom Voice, Voice Design, Voice Cloning
- Clicking each sidebar item switches the main panel
- Custom Voice panel shows Speaker dropdown (grouped by language), Emotion input, Speed radios, Text textarea, Generate button
- Voice Design shows description + text fields
- Voice Cloning shows Saved Voices / Quick Upload tabs
- Output area at bottom shows placeholder text

- [ ] **Step 3: Test Custom Voice generation end-to-end**

1. Select speaker (e.g. Vivian)
2. Leave emotion blank
3. Select Normal speed
4. Type "Hello, this is a test of the Qwen3 TTS system."
5. Click Generate
6. Button should show "Generating…" and a spinner appears in the output area
7. After generation (10-30s): audio player appears with play button and Download link
8. Click play — audio should play in browser
9. Click Download — WAV file downloads

- [ ] **Step 4: Test Voice Design**

1. Click Voice Design in sidebar
2. Enter description: "A calm, deep male voice"
3. Enter text: "The quick brown fox jumps over the lazy dog."
4. Click Generate → verify audio plays

- [ ] **Step 5: Test Voice Cloning — Quick Upload**

1. Click Voice Cloning → Quick Upload tab
2. Drag a WAV/MP3 file onto the drop zone (filename appears)
3. Optionally enter a transcript
4. Enter text to synthesize
5. Click Generate → verify cloned audio plays

- [ ] **Step 6: Test Voice Cloning — Saved Voices**

1. Click Voice Cloning → Saved Voices tab
2. Click "+ Enroll New Voice"
3. Enter name, upload audio, enter transcript → click Save Voice
4. New voice appears in list
5. Click voice to select it (highlighted)
6. Enter text and click Generate → verify it works
7. Click ✕ Remove on the voice → confirm → voice disappears

- [ ] **Step 7: Test error state**

1. In Custom Voice, leave text blank and click Generate → alert appears
2. (Optional) Stop the server mid-generation → error message appears in output area

- [ ] **Step 8: Commit final state**

```bash
git add app.py static/index.html docs/
git commit -m "feat: add Qwen3-TTS web interface (FastAPI + single-page HTML UI)"
```

---

## Summary

| Task | Deliverable | Verification |
|------|------------|-------------|
| 1 | Dependencies installed, server skeleton running | `curl /health` returns `{"status":"ok"}` |
| 2 | Full backend with all endpoints | curl smoke tests pass |
| 3 | Complete SPA frontend | Browser UI works end-to-end for all 3 modes |
