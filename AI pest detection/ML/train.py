import argparse
import json
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from model import create_model

def train_model(data_dir, epochs=10, batch_size=32, img_size=224):
    """Train the pest detection model"""
    
    # Load datasets
    train_dir = Path(data_dir) / 'train'
    val_dir = Path(data_dir) / 'val'
    
    train_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset='training',
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )
    
    val_ds = keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset='validation',
        seed=42,
        image_size=(img_size, img_size),
        batch_size=batch_size
    )
    
    # Get class names
    class_names = train_ds.class_names
    print(f"Classes: {class_names}")
    
    # Save class names
    os.makedirs('models', exist_ok=True)
    with open('models/labels.json', 'w') as f:
        json.dump(class_names, f)
    
    # Create model
    model, base_model = create_model(len(class_names), img_size)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            'models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2
        )
    ]
    
    # Stage 1: Train head
    print("\n=== Stage 1: Training classifier head ===")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Stage 2: Fine-tuning
    print("\n=== Stage 2: Fine-tuning entire model ===")
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs // 2,
        callbacks=callbacks
    )
    
    # Save final model
    model.save('models/final_model.keras')
    print("\n✅ Training complete! Model saved to models/final_model.keras")
    
    return model, class_names

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    
    args = parser.parse_args()
    train_model(args.data_dir, args.epochs, args.batch_size, args.img_size)