import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from training.dataset import load_datasets

train_ds, val_ds, class_names = load_datasets()

print("\nClasses:")
print(class_names)

for images, labels in train_ds.take(1):
    print(f"\nImages Shape : {images.shape}")
    print(f"Labels Shape : {labels.shape}")