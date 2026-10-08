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
