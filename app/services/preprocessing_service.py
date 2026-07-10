import numpy as np
from PIL import Image

from app.config import IMAGE_SIZE


def preprocess_image(image: Image.Image):

    image = image.resize(IMAGE_SIZE)

    image = np.array(image)

    image = image.astype("float32") / 255.0

    image = np.expand_dims(image, axis=0)

    return image