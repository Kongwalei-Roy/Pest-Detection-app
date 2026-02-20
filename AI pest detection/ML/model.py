import tensorflow as tf  # type: ignore
from tensorflow import keras  # type: ignore
from tensorflow.keras import layers, applications  # type: ignore

def create_model(num_classes, input_size=224):
    """Create MobileNetV3 based model for pest classification"""
    
    # Load pretrained MobileNetV3
    base_model = applications.MobileNetV3Small(
        input_shape=(input_size, input_size, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(input_size, input_size, 3))
    
    # Data preprocessing (normalization)
    x = layers.Rescaling(1./255)(inputs)
    
    # Data augmentation (for training)
    x = layers.RandomFlip('horizontal')(x)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model