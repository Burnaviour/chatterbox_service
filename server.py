from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import uvicorn
import io
import soundfile as sf
import torch
import sys

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

from chatterbox.tts_turbo import ChatterboxTurboTTS

app = FastAPI()
model = None


class TTSRequest(BaseModel):
    text: str
    audio_prompt_path: str = None


@app.on_event("startup")
def load_model():
    global model
    print("Loading Chatterbox Turbo model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = ChatterboxTurboTTS.from_pretrained(device=device)
    
    # Pre-load default voice if available
    try:
        print("Loading default voice from record.wav...")
        model.prepare_conditionals("record.wav")
        print("Default voice loaded.")
    except Exception as e:
        print(f"Warning: Could not load default voice 'record.wav': {e}")
        print("You must provide `audio_prompt_path` in requests.")
    
    print("Model loaded.")


@app.post("/generate")
def generate_audio(request: TTSRequest):
    # if request.audio_prompt_path is None:
    #     raise HTTPException(400, "audio_prompt_path is required (zero-shot cloning)")

    try:
        wav = model.generate(
            request.text,
            # audio_prompt_path=request.audio_prompt_path,
            # You can add temperature, top_p etc if the method supports them
        )

        audio_np = wav.cpu().numpy().squeeze()
        if audio_np.ndim != 1:
            raise ValueError(f"Unexpected audio shape: {audio_np.shape}")

        buffer = io.BytesIO()
        sf.write(buffer, audio_np, model.sr, format="WAV", subtype="PCM_16")
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, str(e))

    except Exception as e:
        print(f"Error generating audio: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
