import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
    
from src.dataset_loader import load_dataset
from src.preprocessing import preprocess_audio
from src.augmentation import random_augmentation
from src.config import SAMPLE_RATE

df = load_dataset()

audio_path = df.iloc[0]["path"]

signal = preprocess_audio(audio_path)

augmented = random_augmentation(signal, SAMPLE_RATE)

print(signal.shape)
print(augmented.shape)