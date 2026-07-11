import pickle

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE

X = np.load("dataset/features/X.npy")

y = np.load("dataset/features/y.npy")

X = X.transpose(0, 2, 1)

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

model = load_model(
    "saved_models/saved_model.keras"
)

y_prob = model.predict(X_test)

y_pred = np.argmax(
    y_prob,
    axis=1
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("\nModel Evaluation")

print("-"*30)

print(f"Accuracy : {accuracy:.4f}")

print(f"Precision: {precision:.4f}")

print(f"Recall   : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")


print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred
    )
)

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(10,8))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    "images/confusion_matrix.png"
)

plt.show()

with open(
    "logs/history.pkl",
    "rb"
) as file:

    history = pickle.load(file)

plt.figure(figsize=(8,5))

plt.plot(history["accuracy"])

plt.plot(history["val_accuracy"])

plt.title("Training Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend([
    "Train",
    "Validation"
])

plt.tight_layout()

plt.savefig(
    "images/accuracy.png"
)

plt.show()

plt.figure(figsize=(8,5))

plt.plot(history["loss"])

plt.plot(history["val_loss"])

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend([
    "Train",
    "Validation"
])

plt.tight_layout()

plt.savefig(
    "images/loss.png"
)

plt.show()

plt.figure(figsize=(8,5))

plt.plot(history["loss"])

plt.plot(history["val_loss"])

plt.title("Training Loss")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend([
    "Train",
    "Validation"
])

plt.tight_layout()

plt.savefig(
    "images/loss.png"
)

plt.show()