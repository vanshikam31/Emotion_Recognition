import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.dataset_loader import load_dataset

df = load_dataset()

print(df.head())

print("\nTotal Samples:", len(df))

print("\nEmotion Counts:\n")

print(df["emotion"].value_counts())