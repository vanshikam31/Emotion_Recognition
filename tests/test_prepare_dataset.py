from src.prepare_dataset import prepare_dataset

X, y = prepare_dataset()

print("Feature Shape :", X.shape)

print("Label Shape :", y.shape)

print("\nClasses Saved Successfully.")