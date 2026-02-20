import argparse
from pathlib import Path

import numpy as np
try:
    import tensorflow as tf
except ImportError:
    print("Error: TensorFlow is not installed. Please install it with 'pip install tensorflow'.")
    exit(1)
from sklearn.metrics import classification_report, confusion_matrix

from .utils import load_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_path", type=str, default="ml/final_model.keras")
    parser.add_argument("--labels_path", type=str, default="ml/labels.json")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    test_dir = Path(args.data_dir) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(
            "Missing data/test folder.\n"
            "Create data/test/<class_name>/*.jpg"
        )

    labels = load_labels(args.labels_path)
    model = tf.keras.models.load_model(args.model_path)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        label_mode="int",
        shuffle=False,
    )

    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
    probs = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    acc = float(np.mean(y_pred == y_true))
    print("\n✅ Test Accuracy:", acc)

    print("\n✅ Classification Report:")
    print(classification_report(y_true, y_pred, target_names=labels))

    print("\n✅ Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()
