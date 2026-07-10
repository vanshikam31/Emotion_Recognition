from src.prepare_dataset import prepare_dataset

X, y = prepare_dataset()

print("\nFeature Shape :", X.shape)

print("Label Shape :", y.shape)

print("\nClasses:\n")

print(set(y))