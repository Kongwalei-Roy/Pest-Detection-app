import json
import numpy as np
from PIL import Image
import tensorflow as tf
import logging

logger = logging.getLogger(__name__)

class PestClassifier:
    """TensorFlow Lite classifier for pest detection"""
    
    def __init__(self, model_path: str, labels_path: str, img_size: int = 224):
        """
        Initialize the classifier
        
        Args:
            model_path: Path to .tflite model file
            labels_path: Path to labels.json file
            img_size: Input image size (default: 224)
        """
        self.img_size = img_size
        self.labels_path = labels_path
        self.model_path = model_path
        
        # Load labels
        try:
            with open(labels_path, 'r') as f:
                self.labels = json.load(f)
            logger.info(f"✅ Loaded {len(self.labels)} labels from {labels_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load labels: {e}")
            raise
        
        # Load TFLite model
        try:
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            
            # Get model details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            # Log model info
            logger.info(f"✅ Loaded model from {model_path}")
            logger.info(f"📊 Input shape: {self.input_details[0]['shape']}")
            logger.info(f"📊 Output shape: {self.output_details[0]['shape']}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def preprocess_image(self, image: Image.Image) -> np.ndarray:
        """
        Preprocess image for model input
        
        Args:
            image: PIL Image object
            
        Returns:
            Preprocessed numpy array
        """
        # Resize
        image = image.resize((self.img_size, self.img_size))
        
        # Convert to array and normalize to [0, 1]
        img_array = np.array(image, dtype=np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    
    def predict(self, image: Image.Image, top_k: int = 3) -> list:
        """
        Predict pest class from image
        
        Args:
            image: PIL Image object
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with class and confidence
        """
        try:
            # Preprocess
            input_data = self.preprocess_image(image)
            
            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # Get predictions
            predictions = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
            
            # Get top k indices
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            # Format results
            results = []
            for idx in top_indices:
                results.append({
                    'class': self.labels[idx],
                    'confidence': float(predictions[idx])
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise
    
    def predict_with_metadata(self, image: Image.Image) -> dict:
        """
        Predict with additional metadata
        
        Returns:
            Dictionary with predictions and metadata
        """
        start_time = tf.timestamp()
        
        predictions = self.predict(image)
        
        inference_time = tf.timestamp() - start_time
        
        return {
            'predictions': predictions,
            'metadata': {
                'model': self.model_path,
                'labels': self.labels_path,
                'inference_time': float(inference_time),
                'input_size': self.img_size
            }
        }
    
    def get_model_info(self) -> dict:
        """Get model information"""
        return {
            'model_path': self.model_path,
            'labels_path': self.labels_path,
            'num_classes': len(self.labels),
            'classes': self.labels,
            'input_size': self.img_size,
            'input_details': {
                'shape': self.input_details[0]['shape'].tolist(),
                'dtype': str(self.input_details[0]['dtype'])
            },
            'output_details': {
                'shape': self.output_details[0]['shape'].tolist(),
                'dtype': str(self.output_details[0]['dtype'])
            }
        }