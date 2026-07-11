import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.prepare_dataset import prepare_dataset

X, y = prepare_dataset()

print("Feature Shape :", X.shape)

print("Label Shape :", y.shape)

print("\nClasses Saved Successfully.")