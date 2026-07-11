import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset_loader import load_dataset
from src.preprocessing import preprocess_audio

df = load_dataset()

audio_path = df.iloc[0]["path"]

signal = preprocess_audio(audio_path)

print("Processed Signal Shape:", signal.shape)
print("Total Samples:", len(signal))