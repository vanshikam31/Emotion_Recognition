from pathlib import Path

# Project Root Directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset Paths
RAW_DATASET_PATH = PROJECT_ROOT / "dataset" / "raw" / "audio_speech_actors_01-24"
PROCESSED_DATASET_PATH = PROJECT_ROOT / "dataset" / "processed"

# Model Path
MODEL_PATH = PROJECT_ROOT / "saved_models"

# Logs
LOGS_PATH = PROJECT_ROOT / "logs"

# Images
IMAGES_PATH = PROJECT_ROOT / "images"