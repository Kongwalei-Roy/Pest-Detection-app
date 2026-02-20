import tensorflow as tf
from tensorflow import keras
from keras import layers, applications
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = "data"

def load_and_preprocess_data():
    """Load images from directory structure"""
    
    # Training data with augmentation
    train_ds = keras.preprocessing.image_dataset_from_directory(
        f"{DATA_DIR}/train",
        validation_split=0.2,
        subset="training",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int"
    )
    
    # Validation data
    val_ds = keras.preprocessing.image_dataset_from_directory(
        f"{DATA_DIR}/train",
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        label_mode="int"
    )
    
    # Test data (optional)
    test_ds = None
    if os.path.exists(f"{DATA_DIR}/test"):
        test_ds = keras.preprocessing.image_dataset_from_directory(
            f"{DATA_DIR}/test",
            image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            label_mode="int",
            shuffle=False
        )
    
    # Get class names
    class_names = train_ds.class_names
    print(f"\n📊 Found {len(class_names)} classes:")
    for i, name in enumerate(class_names):
        print(f"  {i}: {name}")
    
    # Save class names
    os.makedirs("models", exist_ok=True)
    with open("models/labels.json", "w") as f:
        json.dump(class_names, f)
    
    # Optimize datasets
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)
    
    return train_ds, val_ds, test_ds, class_names

def create_data_augmentation():
    """Create data augmentation pipeline"""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
        layers.RandomBrightness(0.1),
    ], name="data_augmentation")

def build_model(num_classes):
    """Build MobileNetV3 model"""
    
    # Load pretrained MobileNetV3
    base_model = applications.MobileNetV3Small(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze base model initially
    base_model.trainable = False
    
    # Create augmentation layer
    augmentation = create_data_augmentation()
    
    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Augment data (only during training)
    x = augmentation(inputs)
    
    # Normalize to [0,1]
    x = layers.Rescaling(1./255)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Classification head
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    plt.show()

def train_model():
    """Main training function"""
    
    print("=" * 50)
    print("🚀 Starting Pest Detection Model Training")
    print("=" * 50)
    
    # 1. Load data
    train_ds, val_ds, test_ds, class_names = load_and_preprocess_data()
    num_classes = len(class_names)
    
    # 2. Build model
    print("\n🔨 Building model...")
    model, base_model = build_model(num_classes)
    model.summary()
    
    # 3. Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 4. Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=1
        )
    ]
    
    # 5. Stage 1: Train head only
    print("\n📚 Stage 1: Training classifier head...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Stage 2: Fine-tune entire model
    print("\n🔧 Stage 2: Fine-tuning entire model...")
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
        initial_epoch=history1.epoch[-1] + 1
    )
    
    # 7. Combine histories
    history = {}
    for key in history1.history.keys():
        history[key] = history1.history[key] + history2.history[key]
    
    # 8. Save final model
    model.save('models/final_model.keras')
    print(f"\n✅ Model saved to models/final_model.keras")
    
    # 9. Plot training history
    plot_training_history(history)
    
    # 10. Evaluate on test set if available
    if test_ds:
        print("\n📊 Evaluating on test set...")
        test_loss, test_acc = model.evaluate(test_ds)
        print(f"Test accuracy: {test_acc:.4f}")
    
    # 11. Save as TFLite
    print("\n🔄 Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    
    with open('models/pest_classifier.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite model saved to models/pest_classifier.tflite")
    print(f"\n🎉 Training complete! Models saved in 'models/' directory")

if __name__ == "__main__":
    train_model()