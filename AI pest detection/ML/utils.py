import json
import numpy as np
from PIL import Image

def load_labels(labels_path):
    """Load class labels from JSON file"""
    with open(labels_path, 'r') as f:
        return json.load(f)

def preprocess_image(image_path, img_size=224):
    """Load and preprocess image for prediction"""
    image = Image.open(image_path).convert('RGB')
    image = image.resize((img_size, img_size))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def format_prediction(predictions, labels, top_k=3):
    """Format model predictions into readable format"""
    top_indices = np.argsort(predictions[0])[-top_k:][::-1]
    results = []
    for idx in top_indices:
        results.append({
            'class': labels[idx],
            'confidence': float(predictions[0][idx])
        })
    return results