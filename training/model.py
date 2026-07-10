import tensorflow as tf
from training.augmentation import data_augmentation

NUM_CLASSES = 3

base_model = tf.keras.applications.EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False


def build_model():

    inputs = tf.keras.Input(shape=(224, 224, 3))

    x = data_augmentation(inputs)

    x = tf.keras.applications.efficientnet.preprocess_input(x)

    x = base_model(x, training=False)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    x = tf.keras.layers.Dropout(0.3)(x)

    x = tf.keras.layers.Dense(
        128,
        activation="relu"
    )(x)

    outputs = tf.keras.layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )(x)

    model = tf.keras.Model(inputs, outputs)

    return model