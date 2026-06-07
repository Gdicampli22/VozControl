import librosa
import numpy as np

def extraer_features(audio, sr=16000):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=26)
    mfcc_mean = np.mean(mfcc, axis=1)

    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    rms = np.mean(librosa.feature.rms(y=audio))
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    features = np.hstack([
        mfcc_mean,
        zcr,
        rms,
        spec_centroid
    ])

    features = np.hstack([...])  # lo que ya tenés

    # AJUSTE PARA MODELO SINTÉTICO
    if len(features) < 30:
        features = np.pad(features, (0, 30 - len(features)))
    elif len(features) > 30:
        features = features[:30]



    return features
    