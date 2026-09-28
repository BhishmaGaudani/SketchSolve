"""Train the symbol CNN. Run from the repo root:

    python -m scripts.train

Writes models/symbol_cnn.keras, models/label_map.json and
docs/training_curves.png. Uses only the train and val splits; the test split
is left for scripts/evaluate.py.
"""
import json

import matplotlib
matplotlib.use("Agg")  # draw plots to files, no window
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src.config import CLASSES, DOCS_DIR, LABEL_MAP_PATH, MODEL_PATH, set_seed
from src.data import load_split
from src.model import build_model

MAX_EPOCHS = 50  # an upper limit; EarlyStopping normally ends training sooner
BATCH_SIZE = 64


def save_training_curves(history, path):
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    epochs = range(1, len(history["loss"]) + 1)
    ax_loss.plot(epochs, history["loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="validation")
    ax_loss.set(title="Loss", xlabel="epoch")
    ax_acc.plot(epochs, history["accuracy"], label="train")
    ax_acc.plot(epochs, history["val_accuracy"], label="validation")
    ax_acc.set(title="Accuracy", xlabel="epoch")
    for ax in (ax_loss, ax_acc):
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main():
    set_seed()
    x_train, y_train = load_split("train")
    x_val, y_val = load_split("val")
    print(f"Train: {len(x_train)} images, val: {len(x_val)} images, {len(CLASSES)} classes")

    # Rare classes (like ÷) get a bigger weight in the loss, so the model
    # doesn't just ignore them. "balanced" weights (n_samples / (n_classes *
    # class_count)) made the model over-guess rare classes; their square root
    # is a milder boost that worked better on the validation set.
    labels = np.arange(len(CLASSES))
    weights = np.sqrt(compute_class_weight("balanced", classes=labels, y=y_train))
    class_weight = dict(zip(labels.tolist(), weights.tolist()))

    model = build_model(len(CLASSES))
    model.summary()

    # Stop once validation loss hasn't improved for 5 epochs, and roll back to
    # the epoch where it was lowest.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)

    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[early_stopping],
    )

    MODEL_PATH.parent.mkdir(exist_ok=True)
    model.save(MODEL_PATH)
    label_map = {str(i): symbol for i, (symbol, _) in enumerate(CLASSES)}
    LABEL_MAP_PATH.write_text(json.dumps(label_map, ensure_ascii=False, indent=2))

    DOCS_DIR.mkdir(exist_ok=True)
    save_training_curves(history.history, DOCS_DIR / "training_curves.png")

    best = int(np.argmin(history.history["val_loss"]))
    print(f"Epochs run: {len(history.history['loss'])}, best epoch: {best + 1}")
    print(f"Best epoch val_loss: {history.history['val_loss'][best]:.4f}, "
          f"val_accuracy: {history.history['val_accuracy'][best]:.4f}")
    print(f"Saved {MODEL_PATH} and {LABEL_MAP_PATH}")


if __name__ == "__main__":
    main()
