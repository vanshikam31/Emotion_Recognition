import numpy as np
from tqdm import tqdm

from src.dataset_loader import load_dataset
from src.preprocessing import preprocess_audio
from src.feature_extraction import extract_mfcc

def prepare_dataset():

    df = load_dataset()

    X = []
    y = []

    print(f"\nTotal Audio Files : {len(df)}\n")

    for _, row in tqdm(df.iterrows(), total=len(df)):

        signal = preprocess_audio(row["path"])

        mfcc = extract_mfcc(signal)

        X.append(mfcc)

        y.append(row["emotion"])

    X = np.array(X)

    y = np.array(y)

    return X, y


