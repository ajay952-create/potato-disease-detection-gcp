import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from training.dataset import load_datasets


MODEL_PATH = "models/potato_disease_model.keras"


def main():

    print("=" * 60)
    print("Loading Model...")
    print("=" * 60)

    model = tf.keras.models.load_model(MODEL_PATH)

    print("Model loaded successfully.\n")

    print("Loading Validation Dataset...")

    _, val_ds, class_names = load_datasets()

    y_true = []
    y_pred = []

    print("Generating Predictions...\n")

    for images, labels in val_ds:

        predictions = model.predict(images, verbose=0)

        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels.numpy(), axis=1)

        y_pred.extend(predicted_classes)
        y_true.extend(true_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(
        y_true,
        y_pred,
        average="weighted"
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted"
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted"
    )

    cm = confusion_matrix(y_true, y_pred)

    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)

    print(f"\nAccuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nClassification Report")
    print("-" * 60)

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=class_names
        )
    )

    print("\nConfusion Matrix")
    print("-" * 60)

    print(cm)


if __name__ == "__main__":
    main()