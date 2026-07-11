import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.predict import predict_emotion
import joblib

audio_path = "dataset/raw/audio_speech_actors_01-24/Actor_01/03-01-01-01-01-01-01.wav"

emotion, confidence = predict_emotion(audio_path)

print("\nPredicted Emotion:", emotion)

print("\nConfidence Scores")
print("-" * 30)

encoder = joblib.load("saved_models/label_encoder.pkl")

for emotion_name, score in zip(encoder.classes_, confidence):
    print(f"{emotion_name:10} : {score:.2f}%")