from pathlib import Path


def create_directory(path):
    """
    Create a directory if it does not already exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)