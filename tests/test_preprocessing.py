from src.dataset_loader import load_dataset
from src.preprocessing import preprocess_audio

df = load_dataset()

audio_path = df.iloc[0]["path"]

signal = preprocess_audio(audio_path)

print("Processed Signal Shape:", signal.shape)
print("Total Samples:", len(signal))