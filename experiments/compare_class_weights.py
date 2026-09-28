"""One-off experiment: which class weighting should training use?

Trains the same model three times (no weights, square-root weights, balanced
weights) and compares them on the VALIDATION set, never the test set.

I ran this once, when × and x were still two separate classes (16 classes,
with "times" at index 12, "÷" at 13 and "X" at 15). The results are what
made me merge × and x and pick square-root weights; the raw output is in
docs/class_weight_experiment.md. It won't run as-is against the current
15-class config, because the class indices below no longer exist.

    PYTHONPATH=. python experiments/compare_class_weights.py
"""
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras

from src.config import CLASSES, set_seed
from src.data import load_split
from src.model import build_model

TIMES, DIVIDE, X = 12, 13, 15  # class indices in the 16-class setup

x_tr, y_tr = load_split("train")
x_va, y_va = load_split("val")
labels = np.arange(len(CLASSES))
balanced = compute_class_weight("balanced", classes=labels, y=y_tr)
schemes = {
    "none": None,
    "sqrt": dict(zip(labels.tolist(), np.sqrt(balanced).tolist())),
    "balanced": dict(zip(labels.tolist(), balanced.tolist())),
}

for name, class_weight in schemes.items():
    set_seed()
    model = build_model(len(CLASSES))
    model.fit(x_tr, y_tr, validation_data=(x_va, y_va), epochs=50, batch_size=64,
              class_weight=class_weight, verbose=0,
              callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=5,
                                                       restore_best_weights=True)])
    pred = model.predict(x_va, verbose=0).argmax(1)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_va, pred, labels=labels, zero_division=0)
    other = ~np.isin(y_va, [TIMES, X])
    print(f"{name:9s} val_acc={np.mean(pred == y_va):.4f}  "
          f"×: P={precision[TIMES]:.3f} R={recall[TIMES]:.3f}  "
          f"x: P={precision[X]:.3f} R={recall[X]:.3f}  ÷: R={recall[DIVIDE]:.3f}  "
          f"other14_acc={np.mean(pred[other] == y_va[other]):.4f}", flush=True)
