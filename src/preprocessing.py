import librosa
import numpy as np

# Target Sampling Rate
SAMPLE_RATE = 22050

# Fixed audio duration (seconds)
DURATION = 3

# Total samples
SAMPLES = SAMPLE_RATE * DURATION

def load_audio(audio_path):
    """
    Load an audio file.
    """

    signal, sr = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    return signal, sr


def trim_silence(signal):
    """
    Remove leading and trailing silence.
    """

    trimmed_signal, _ = librosa.effects.trim(signal)

    return trimmed_signal


def normalize_audio(signal):
    """
    Normalize audio amplitude.
    """

    max_value = np.max(np.abs(signal))

    if max_value > 0:
        signal = signal / max_value

    return signal


def pad_or_truncate(signal):
    """
    Make every audio exactly 3 seconds.
    """

    if len(signal) < SAMPLES:

        padding = SAMPLES - len(signal)

        signal = np.pad(signal, (0, padding))

    else:

        signal = signal[:SAMPLES]

    return signal


def preprocess_audio(audio_path):
    """
    Complete preprocessing pipeline.
    """

    signal, _ = load_audio(audio_path)

    signal = trim_silence(signal)

    signal = normalize_audio(signal)

    signal = pad_or_truncate(signal)

    return signal