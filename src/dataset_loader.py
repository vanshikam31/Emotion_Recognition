from pathlib import Path
import pandas as pd

from src.config import RAW_DATASET_PATH


# Emotion mapping for RAVDESS dataset
EMOTION_MAP = {
    "01": "Neutral",
    "02": "Calm",
    "03": "Happy",
    "04": "Sad",
    "05": "Angry",
    "06": "Fear",
    "07": "Disgust",
    "08": "Surprise"
}


def load_dataset():
    """
    Scan the RAVDESS dataset and create a DataFrame
    containing file paths and corresponding emotion labels.
    """

    data = []

    # Traverse all Actor folders
    for actor_folder in RAW_DATASET_PATH.iterdir():

        if not actor_folder.is_dir():
            continue

        # Traverse all audio files
        for audio_file in actor_folder.glob("*.wav"):

            parts = audio_file.stem.split("-")

            emotion_code = parts[2]

            emotion = EMOTION_MAP.get(emotion_code)

            data.append({
                "path": str(audio_file),
                "emotion": emotion
            })

    dataframe = pd.DataFrame(data)

    return dataframe