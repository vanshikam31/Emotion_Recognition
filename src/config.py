from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset Paths
RAW_DATASET_PATH = PROJECT_ROOT / "dataset" / "raw" / "audio_speech_actors_01-24"
PROCESSED_DATASET_PATH = PROJECT_ROOT / "dataset" / "processed"

# Model
MODEL_PATH = PROJECT_ROOT / "saved_models"

# Logs
LOGS_PATH = PROJECT_ROOT / "logs"

# Images
IMAGES_PATH = PROJECT_ROOT / "images"

# Audio Configuration
SAMPLE_RATE = 22050
DURATION = 3
SAMPLES = SAMPLE_RATE * DURATION
N_MFCC = 40