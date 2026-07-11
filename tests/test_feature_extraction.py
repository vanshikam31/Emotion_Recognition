import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset_loader import load_dataset
from src.preprocessing import preprocess_audio
from src.feature_extraction import extract_all_features

df = load_dataset()

audio_path = df.iloc[0]["path"]

signal = preprocess_audio(audio_path)

features = extract_all_features(signal)

print("\nExtracted Features:\n")

for feature_name, feature_value in features.items():
    print(f"{feature_name}: {feature_value.shape}")