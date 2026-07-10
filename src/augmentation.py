import numpy as np
import librosa

def add_noise(signal, noise_factor=0.005):
    """
    Add random Gaussian noise.
    """
    noise = np.random.randn(len(signal))
    augmented = signal + noise_factor * noise
    return augmented

def pitch_shift(signal, sample_rate, steps=2):
    """
    Shift pitch without changing duration.
    """
    return librosa.effects.pitch_shift(
        signal,
        sr=sample_rate,
        n_steps=steps
    )



def time_stretch(signal, rate=1.1):
    """
    Change speaking speed.
    """
    return librosa.effects.time_stretch(signal, rate=rate)


def volume_scale(signal, factor=1.2):
    """
    Increase or decrease loudness.
    """
    return signal * factor


import random

def random_augmentation(signal, sample_rate):
    """
    Apply one random augmentation.
    """

    operations = [
        lambda x: add_noise(x),
        lambda x: pitch_shift(x, sample_rate),
        lambda x: time_stretch(x),
        lambda x: volume_scale(x)
    ]

    operation = random.choice(operations)

    return operation(signal)