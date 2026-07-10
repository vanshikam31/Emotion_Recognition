import librosa
import numpy as np

def extract_mfcc(signal, sample_rate):
    """
    Extract MFCC features.
    """

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=sample_rate,
        n_mfcc=40
    )

    return mfcc


def extract_chroma(signal, sample_rate):
    """
    Extract Chroma features.
    """

    stft = np.abs(librosa.stft(signal))

    chroma = librosa.feature.chroma_stft(
        S=stft,
        sr=sample_rate
    )

    return chroma

def extract_mel(signal, sample_rate):
    """
    Extract Mel Spectrogram.
    """

    mel = librosa.feature.melspectrogram(
        y=signal,
        sr=sample_rate
    )

    return mel


def extract_zcr(signal):
    """
    Extract Zero Crossing Rate.
    """

    zcr = librosa.feature.zero_crossing_rate(signal)

    return zcr

def extract_rms(signal):
    """
    Extract RMS Energy.
    """

    rms = librosa.feature.rms(y=signal)

    return rms

def extract_centroid(signal, sample_rate):
    """
    Extract Spectral Centroid.
    """

    centroid = librosa.feature.spectral_centroid(
        y=signal,
        sr=sample_rate
    )

    return centroid


from src.preprocessing import SAMPLE_RATE


def extract_features(signal):
    """
    Extract all features.
    """

    features = {
        "mfcc": extract_mfcc(signal, SAMPLE_RATE),
        "chroma": extract_chroma(signal, SAMPLE_RATE),
        "mel": extract_mel(signal, SAMPLE_RATE),
        "zcr": extract_zcr(signal),
        "rms": extract_rms(signal),
        "centroid": extract_centroid(signal, SAMPLE_RATE)
    }

    return features