import numpy as np
import tensorflow as tf

from app.config import MODEL_PATH
from app.config import CLASS_NAMES

model = None


def load_model():
    global model

    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH)

    return model


def predict(image):

    model = load_model()

    predictions = model.predict(image)

    predicted_index = np.argmax(predictions[0])

    confidence = float(np.max(predictions[0]))

    return {
        "prediction": CLASS_NAMES[predicted_index],
        "confidence": confidence
    }