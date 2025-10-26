import base64
import io

import librosa
from fastapi import FastAPI
from pydantic import BaseModel

from extractor.melody_extractor import MelodyExtractor
from extractor.audio_utils import waveform_to_wav_bytes


app = FastAPI()


class ConvertMelodyRequest(BaseModel):
    input_audio: str  # Base64 encoded audio data


class ConvertMelodyResponse(BaseModel):
    output_audio: str  # Base64 encoded audio data with extracted melody


@app.post("/convert-melody", response_model=ConvertMelodyResponse)
async def convert_melody(request: ConvertMelodyRequest):
    audio_bytes = base64.b64decode(request.input_audio)
    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
    extractor = MelodyExtractor()
    if sr != extractor.params.sample_rate:
        audio = librosa.resample(
            audio, orig_sr=sr, target_sr=extractor.params.sample_rate
        )
    f0, events = extractor.extract_melody(audio)
    output_audio = extractor.events_to_piano_wav(events)
    wav_bytes = waveform_to_wav_bytes(output_audio, extractor.params.sample_rate)
    output_base64 = base64.b64encode(wav_bytes).decode("utf-8")
    return ConvertMelodyResponse(output_audio=output_base64)
