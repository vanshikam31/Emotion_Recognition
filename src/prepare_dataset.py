import numpy as np
from tqdm import tqdm

from src.dataset_loader import load_dataset
from src.preprocessing import preprocess_audio
from src.feature_extraction import extract_mfcc


import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder

from src.config import FEATURE_PATH


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

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y)

    # Save features
    np.save(FEATURE_PATH / "X.npy", X)

    np.save(FEATURE_PATH / "y.npy", y)

    joblib.dump(
        label_encoder,
        FEATURE_PATH / "label_encoder.pkl"
    )

    return X, y