# -*- coding: utf-8 -*-
from TTS.api import TTS
import whisper, os
def synthesize_en_text(text, out_wav, tts_model="tts_models/en/ljspeech/tacotron2-DDC"):
    os.makedirs(os.path.dirname(out_wav), exist_ok=True)
    TTS(tts_model).tts_to_file(text=text, file_path=out_wav)
def transcribe_hu(wav_path, model_name="large-v2"):
    model = whisper.load_model(model_name)
    return model.transcribe(wav_path, language="hu").get("text","").strip()
