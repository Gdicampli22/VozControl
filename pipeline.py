import librosa
import numpy as np
from features import extraer_features

TARGET_SR = 16000
TARGET_LEN = 16000

def procesar_audio(path, gain=1.0):
    audio, sr = librosa.load(path, sr=TARGET_SR)

    # Aplicar ganancia
    audio = audio * gain

    # Trimming
    audio, _ = librosa.effects.trim(audio, top_db=25)

    # Normalización
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    # Longitud fija
    audio = librosa.util.fix_length(audio, size=TARGET_LEN)

    # Features
    features = extraer_features(audio)

    return features.reshape(1, -1), audio