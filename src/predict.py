import joblib
import librosa
import numpy as np

from tensorflow.keras.models import load_model

from src.feature_extraction import extract_features


model = load_model(
    "saved_models/saved_model.keras"
)

label_encoder = joblib.load(
    "saved_models/label_encoder.pkl"
)

def predict_emotion(audio_path):

    features = extract_features(audio_path)

    features = np.expand_dims(features, axis=0)

    features = features.transpose(0, 2, 1)

    prediction = model.predict(features)

    predicted_index = np.argmax(prediction)

    predicted_emotion = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    confidence = prediction[0] * 100

    return predicted_emotion, confidence

