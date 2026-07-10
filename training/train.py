import tensorflow as tf

from training.dataset import load_datasets
from training.model import build_model


def main():

    train_ds, val_ds, class_names = load_datasets()

    model = build_model()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-3
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            filepath="models/best_model.keras",
            monitor="val_accuracy",
            save_best_only=True
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks
    )

    model.save(
        "models/potato_disease_model.keras"
    )

    print("\nTraining completed successfully.")
    print("Model saved to models/potato_disease_model.keras")


if __name__ == "__main__":
    main()