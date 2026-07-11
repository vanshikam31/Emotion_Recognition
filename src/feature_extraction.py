"""
feature_extraction.py

This module extracts audio features for
Speech Emotion Recognition.
"""

import librosa
import numpy as np

from src.config import SAMPLE_RATE, N_MFCC


def extract_mfcc(signal):
    """
    Extract MFCC features.

    Returns:
        numpy.ndarray
    """

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    return mfcc


def extract_chroma(signal):
    """
    Extract Chroma Features.

    Returns:
        numpy.ndarray
    """

    stft = np.abs(librosa.stft(signal))

    chroma = librosa.feature.chroma_stft(
        S=stft,
        sr=SAMPLE_RATE
    )

    return chroma


def extract_mel(signal):
    """
    Extract Mel Spectrogram.

    Returns:
        numpy.ndarray
    """

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=SAMPLE_RATE
    )

    return mel


def extract_zcr(signal):
    """
    Extract Zero Crossing Rate.

    Returns:
        numpy.ndarray
    """

    zcr = librosa.feature.zero_crossing_rate(signal)

    return zcr


def extract_rms(signal):
    """
    Extract RMS Energy.

    Returns:
        numpy.ndarray
    """

    rms = librosa.feature.rms(y=signal)

    return rms


def extract_spectral_centroid(signal):
    """
    Extract Spectral Centroid.

    Returns:
        numpy.ndarray
    """

    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=SAMPLE_RATE
    )

    return centroid


def extract_all_features(signal):
    """
    Extract all audio features.

    Returns:
        dict
    """

    features = {
        "mfcc": extract_mfcc(signal),
        "chroma": extract_chroma(signal),
        "mel": extract_mel(signal),
        "zcr": extract_zcr(signal),
        "rms": extract_rms(signal),
        "spectral_centroid": extract_spectral_centroid(signal)
    }

    return features

def extract_features(audio_path):
    """
    Extract MFCC features for prediction.

    Returns:
        numpy.ndarray (40, 130)
    """

    signal, _ = librosa.load(
        audio_path,
        sr=SAMPLE_RATE
    )

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCC
    )

    # Pad or trim to fixed length
    if mfcc.shape[1] < 130:

        pad_width = 130 - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )

    else:

        mfcc = mfcc[:, :130]

    return mfcc