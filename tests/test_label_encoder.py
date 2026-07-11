import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

import joblib

encoder = joblib.load("saved_models/label_encoder.pkl")

print(encoder.classes_)