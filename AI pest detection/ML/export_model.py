import tensorflow as tf
import json
import numpy as np

def export_to_tflite(model_path='models/final_model.keras', 
                     output_path='models/pest_classifier.tflite'):
    """Export Keras model to TFLite"""
    
    print(f"📤 Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Optimize for size and speed
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Representative dataset for quantization (optional but recommended)
    def representative_dataset():
        # Load a few sample images for calibration
        for _ in range(100):
            yield [np.random.randn(1, 224, 224, 3).astype(np.float32)]
    
    # Uncomment for full int8 quantization (smallest size)
    # converter.representative_dataset = representative_dataset
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    
    print("🔄 Converting to TFLite...")
    tflite_model = converter.convert()
    
    # Save
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    size_mb = len(tflite_model) / (1024 * 1024)
    print(f"✅ Saved to {output_path} ({size_mb:.2f} MB)")

def test_tflite_model(model_path='models/pest_classifier.tflite', 
                      labels_path='models/labels.json'):
    """Test the TFLite model"""
    
    # Load labels
    with open(labels_path, 'r') as f:
        labels = json.load(f)
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    
    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n📊 Model Details:")
    print(f"  Input shape: {input_details[0]['shape']}")
    print(f"  Input type: {input_details[0]['dtype']}")
    print(f"  Output shape: {output_details[0]['shape']}")
    print(f"  Output type: {output_details[0]['dtype']}")
    print(f"  Classes: {labels}")
    
    # Test with random input
    test_input = np.random.randn(1, 224, 224, 3).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\n✅ Test inference successful!")
    print(f"  Output shape: {output.shape}")
    print(f"  Prediction: {labels[np.argmax(output[0])]}")

if __name__ == "__main__":
    export_to_tflite()
    test_tflite_model()