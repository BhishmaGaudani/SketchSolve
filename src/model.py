"""The CNN that classifies one 28x28 symbol image."""
from tensorflow import keras
from tensorflow.keras import layers


def build_augmentation() -> keras.Sequential:
    """Small random changes to training images, so the model sees more variety.

    These layers only change images during training; at prediction time they
    pass images through untouched.

    Deliberately small, because some symbols differ only by orientation:
    rotating "+" by 45 degrees gives "×", and rotating "-" by 90 gives "1".
    No flips either: a mirrored "2" or "7" is not a valid digit.
    """
    return keras.Sequential([
        layers.RandomRotation(10 / 360),        # up to ±10 degrees
        layers.RandomTranslation(0.1, 0.1),     # up to ±10% (~3 px) shift
        layers.RandomZoom(0.1),                 # up to ±10% size change
    ], name="augmentation")


def build_model(num_classes: int) -> keras.Sequential:
    """Same shape as the original MNIST CNN, plus augmentation and Dropout."""
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        build_augmentation(),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),  # randomly zero 25% of features during training

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model
