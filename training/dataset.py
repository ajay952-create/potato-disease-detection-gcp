import tensorflow as tf
from pathlib import Path

# Configuration
DATA_DIR = Path("data/raw/PlantVillage")

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

AUTOTUNE = tf.data.AUTOTUNE


def load_datasets():
    """
    Create train and validation datasets.
    """

    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical"
    )

    class_names = train_ds.class_names

    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names