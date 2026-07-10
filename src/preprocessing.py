"""
preprocessing.py

This module performs audio preprocessing for the
Speech Emotion Recognition project.
"""

import librosa
import numpy as np

from src.config import SAMPLE_RATE, SAMPLES


def load_audio(audio_path):
    """
    Load an audio file.

    Parameters:
        audio_path (str): Path to the audio file.

    Returns:
        tuple:
            signal (numpy.ndarray): Audio signal
            sample_rate (int): Sampling rate
    """

    signal, sample_rate = librosa.load(
        audio_path,
        sr=SAMPLE_RATE,
        mono=True
    )

    return signal, sample_rate


def trim_silence(signal):
    """
    Remove leading and trailing silence.

    Parameters:
        signal (numpy.ndarray)

    Returns:
        numpy.ndarray
    """

    trimmed_signal, _ = librosa.effects.trim(signal)

    return trimmed_signal


def normalize_audio(signal):
    """
    Normalize the audio amplitude.

    Parameters:
        signal (numpy.ndarray)

    Returns:
        numpy.ndarray
    """

    max_value = np.max(np.abs(signal))

    if max_value > 0:
        signal = signal / max_value

    return signal


def pad_or_truncate(signal):
    """
    Make every audio sample the same length.

    Short audio -> Zero Padding
    Long audio -> Truncation

    Returns:
        numpy.ndarray
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

    Steps:
    1. Load audio
    2. Trim silence
    3. Normalize audio
    4. Pad/Truncate

    Returns:
        numpy.ndarray
    """

    signal, _ = load_audio(audio_path)

    signal = trim_silence(signal)

    signal = normalize_audio(signal)

    signal = pad_or_truncate(signal)

    return signal