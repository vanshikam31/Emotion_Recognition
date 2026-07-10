from src.dataset_loader import load_dataset

df = load_dataset()

print(df.head())

print("\nTotal Samples:", len(df))

print("\nEmotion Counts:\n")

print(df["emotion"].value_counts())