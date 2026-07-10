from src.dataset_loader import load_dataset
from src.preprocessing import preprocess_audio

df = load_dataset()

sample_audio = df.iloc[0]["path"]

signal = preprocess_audio(sample_audio)

print("Processed signal shape:", signal.shape)
print("Total Samples:", len(signal))