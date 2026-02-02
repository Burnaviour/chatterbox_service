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

from chatterbox.tts import ChatterboxTTS

app = FastAPI()
model = None


class TTSRequest(BaseModel):
    text: str


@app.on_event("startup")
def load_model():
    global model
    print("Loading Chatterbox model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = ChatterboxTTS.from_pretrained(device=device)
    print("Model loaded.")


@app.post("/generate")
def generate_audio(request: TTSRequest):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Generate Returns a Tensor usually (1, Time) or (Time,)
        wav = model.generate(request.text)
        print(f"Generated WAV shape: {wav.shape}")

        # Ensure CPU and Numpy
        audio_np = wav.cpu().numpy()

        # Squeeze to 1D if it's (1, N) -> (N,)
        if len(audio_np.shape) == 2 and audio_np.shape[0] == 1:
            audio_np = audio_np.squeeze(0)

        print(f"Final numpy shape: {audio_np.shape}")

        # Scale float32 (-1 to 1) to int16 for compatibility if needed,
        # but soundfile/scipy usually handle float32 nicely.
        # Let's stick to soundfile but with correct 1D shape.

        buffer = io.BytesIO()
        sf.write(buffer, audio_np, model.sr, format="WAV")
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        print(f"Error generating audio: {e}")
        import traceback

        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
