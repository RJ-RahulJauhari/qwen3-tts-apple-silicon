# Qwen3-TTS Web Interface Design

**Date:** 2026-10-08  
**Status:** Approved

---

## Overview

Add a local web interface to the existing Qwen3-TTS CLI app. Users launch a FastAPI server (`python app.py`) and open `localhost:7860` in their browser. The UI mirrors all three CLI modes — Custom Voice, Voice Design, and Voice Cloning — with a sidebar layout, non-blocking generation, and inline audio playback.

---

## Architecture

```
app.py              ← FastAPI server (all backend logic)
static/
  index.html        ← single-page UI (HTML + CSS + JS, no build step)
outputs/            ← existing; generated audio saved here
voices/             ← existing; saved voice enrollments
models/             ← existing; model weights
```

The existing `main.py` functions (`get_smart_path`, `save_audio_file`, `load_model`, `generate_audio`, etc.) are imported and reused by `app.py`. No logic is duplicated.

---

## Backend (`app.py`)

### Server
- FastAPI on `localhost:7860`
- Serves `static/index.html` at `GET /`
- Serves `outputs/` directory at `GET /audio/{path}` for browser playback
- TTS generation runs in a `ThreadPoolExecutor` (mlx is not async-compatible)
- In-memory job store: `dict[job_id → {status, audio_url, error}]`

### Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/` | Serve the SPA |
| `POST` | `/generate` | Start a TTS job; returns `{job_id}` immediately |
| `GET` | `/status/{job_id}` | Returns `{status: pending\|done\|error, audio_url?, error?}` |
| `GET` | `/voices` | Returns list of enrolled voice names |
| `POST` | `/voices/enroll` | Multipart: `name`, `audio` (file), `transcript` (text) |
| `DELETE` | `/voices/{name}` | Remove a saved voice |
| `GET` | `/audio/{path:path}` | Stream a WAV file from outputs/ |

### Generate request body

```json
{
  "mode": "custom" | "design" | "clone_saved" | "clone_upload",
  "text": "...",
  "speaker": "Vivian",          // custom only
  "emotion": "Normal tone",     // custom only
  "speed": 1.0,                 // custom only (0.8 | 1.0 | 1.3)
  "voice_description": "...",   // design only
  "voice_name": "Boss",         // clone_saved only
  "ref_audio_b64": "...",       // clone_upload only (base64 WAV)
  "ref_transcript": "..."       // clone_upload only
}
```

Generation runs in background; client polls `/status/{job_id}`.

---

## Frontend (`static/index.html`)

Single HTML file with inline CSS and JS. No framework, no build step.

### Layout

```
┌──────────────┬────────────────────────────────────┐
│  🎙 Qwen3    │                                    │
│  TTS         │  [Active Mode Title]               │
│──────────────│                                    │
│ ▶ Custom     │  (mode-specific form fields)       │
│   Voice      │                                    │
│              │  Text to synthesize:               │
│   Voice      │  ┌──────────────────────────────┐ │
│   Design     │  │                              │ │
│              │  └──────────────────────────────┘ │
│   Voice      │                                    │
│   Cloning    │  [ Generate ]                      │
│              │                                    │
│──────────────│────────────────────────────────────│
│  Port: 7860  │  ▶ ──────── 0:00   [↓ Download]   │
└──────────────┴────────────────────────────────────┘
```

### Custom Voice panel
- Speaker dropdown grouped by language (English / Chinese / Japanese / Korean)
- Emotion text input with placeholder examples
- Speed radio: Normal (1.0) / Fast (1.3) / Slow (0.8)
- Text textarea
- Generate button

### Voice Design panel
- "Describe the voice" textarea
- Text textarea
- Generate button

### Voice Cloning panel
- Sub-tabs: **Saved Voices** | **Quick Upload**
- **Saved Voices:** fetches `GET /voices`, shows list with radio select; Enroll button opens inline form (name + audio file + transcript)
- **Quick Upload:** drag-and-drop or file-pick for reference audio, optional transcript textarea
- Text textarea (shared)
- Generate button

### Generation UX
1. Button changes to "Generating…" + spinner; form stays interactive
2. Client polls `/status/{job_id}` every 500ms
3. On `done`: audio player appears at bottom with play/pause + progress bar + Download button; filename shown
4. On `error`: error message shown in red below the form

---

## Error Handling

- Model not found → 404 with message; UI shows "Model files not found in models/"
- Generation failure → job status `error` with message from exception
- Voice enroll: invalid audio format → 400; ffmpeg converts non-WAV automatically (reuses `convert_audio_if_needed`)
- Invalid voice name for clone → 404

---

## Dependencies

New packages needed:
- `fastapi`
- `uvicorn`
- `python-multipart` (for file uploads)

All other logic reuses existing `main.py` imports and functions.

---

## Out of Scope

- Authentication (local only)
- Concurrent multi-user generation (single ThreadPoolExecutor worker to avoid memory pressure)
- History / session persistence beyond what already exists in `outputs/`
- 0.6B Lite models (not downloaded; commented out in MODELS)
