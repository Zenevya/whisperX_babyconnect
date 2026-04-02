import numpy as np

try:
    import librosa
except Exception:
    librosa = None

try:
    import noisereduce as nr
except Exception:
    nr = None


def reduce_noise(audio, sample_rate=16000):
    if nr is None:
        return audio

    try:
        return nr.reduce_noise(
            y=audio,
            sr=sample_rate,
            stationary=True,
            prop_decrease=0.75,
        )
    except Exception:
        return audio


def normalize_audio(audio):
    if len(audio) == 0:
        return audio

    if librosa is not None:
        try:
            return librosa.util.normalize(audio)
        except Exception:
            pass

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def compress_audio(audio):
    if len(audio) == 0:
        return audio
    return np.tanh(audio * 2.0) * 0.8