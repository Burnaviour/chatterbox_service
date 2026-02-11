from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn
import io
import soundfile as sf
import torch
import os
import traceback

# --- MONKEYPATCH FIX FOR BROKEN PERTH LIBRARY ---
try:
    import perth

    print(f"Monkeypatching perth.PerthImplicitWatermarker (Current: {perth.PerthImplicitWatermarker})")
    if perth.PerthImplicitWatermarker is None:
        print("Using DummyWatermarker instead.")
        perth.PerthImplicitWatermarker = perth.DummyWatermarker
except ImportError:
    print("Could not import perth for monkeypatching. Proceeding anyway.")
# ------------------------------------------------

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

app = FastAPI()
multilingual_model: ChatterboxMultilingualTTS | None = None


def _select_device() -> str:
    forced = (os.getenv("CHATTERBOX_DEVICE") or "").strip().lower()
    if forced in {"cpu", "cuda"}:
        return forced

    if not torch.cuda.is_available():
        return "cpu"

    # CUDA may be "available" but the installed torch build might not include
    # kernels for this GPU (e.g. RTX 5090 sm_120) => "no kernel image" at runtime.
    try:
        _ = torch.zeros(1, device="cuda")
        return "cuda"
    except Exception as e:
        print(f"CUDA init/test failed ({type(e).__name__}: {e}); falling back to CPU")
        return "cpu"


class TTSRequest(BaseModel):
    text: str
    language_id: str = "en"  # e.g. "en" or "de"
    audio_prompt_path: str | None = None


@app.on_event("startup")
def load_model():
    global multilingual_model
    print("Loading Chatterbox Multilingual model...")
    device = _select_device()
    print(f"Using device: {device}")

    multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    supported = multilingual_model.get_supported_languages()
    print(f"Supported languages: {', '.join(sorted(supported.keys()))}")
    print("Model loaded.")


@app.post("/generate")
def generate_audio(request: TTSRequest):
    try:
        if multilingual_model is None:
            raise RuntimeError("Model not loaded")

        language_id = (request.language_id or "de").strip().lower()
        if language_id not in {"en", "de"}:
            raise HTTPException(status_code=400, detail="Only 'en' and 'de' are supported by this API")

        wav = multilingual_model.generate(
            request.text,
            language_id=language_id,
            audio_prompt_path='record.wav',cfg_weight=0.3,exaggeration=0.7
        )

        audio_np = wav.cpu().numpy().squeeze()
        if audio_np.ndim != 1:
            raise ValueError(f"Unexpected audio shape: {audio_np.shape}")

        buffer = io.BytesIO()
        sf.write(buffer, audio_np, multilingual_model.sr, format="WAV", subtype="PCM_16")
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        print(f"Error generating audio: {e}")
        traceback.print_exc()
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
