import numpy as np

from sklearn.model_selection import train_test_split

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
    TensorBoard
)

from src.config import *
from src.model import build_model

X = np.load("dataset/features/X.npy")

y = np.load("dataset/features/y.npy")

print("Original Shape:", X.shape)

X = X.transpose(0, 2, 1)

print("New Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Training:", X_train.shape)

print("Testing:", X_test.shape)

model = build_model()

model.summary()

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True,
    verbose=1
)

checkpoint = ModelCheckpoint(
    filepath="saved_models/saved_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    verbose=1
)

tensorboard = TensorBoard(
    log_dir="logs",     
    histogram_freq=1
)

callbacks = [
    early_stopping,
    checkpoint,
    reduce_lr,
    tensorboard
]

history = model.fit(
    X_train,
    y_train,
    validation_split=VALIDATION_SPLIT,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

import pickle

with open("logs/history.pkl", "wb") as file:
    pickle.dump(history.history, file)