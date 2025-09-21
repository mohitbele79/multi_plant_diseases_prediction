
from PIL import Image
import numpy as np

def preprocess_image(image_path):
    """
    Preprocess the image for prediction.
    """
    image = Image.open(image_path).resize((224, 224))  # Resize to model's input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

