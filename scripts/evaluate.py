"""Evaluate the trained model on the held-out test split. Run from the repo root:

    python -m scripts.evaluate

Writes docs/confusion_matrix.png and docs/test_metrics.md.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tensorflow import keras

from src.config import DOCS_DIR, LABEL_MAP_PATH, MODEL_PATH
from src.data import load_split


def main():
    model = keras.models.load_model(MODEL_PATH)
    label_map = json.loads(LABEL_MAP_PATH.read_text())
    symbols = [label_map[str(i)] for i in range(len(label_map))]

    x_test, y_test = load_split("test")
    y_pred = model.predict(x_test, verbose=0).argmax(axis=1)

    accuracy = (y_pred == y_test).mean()
    report = classification_report(y_test, y_pred, target_names=symbols, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    # The most frequent mistakes: largest off-diagonal cells.
    off_diagonal = cm.copy()
    np.fill_diagonal(off_diagonal, 0)
    top = np.dstack(np.unravel_index(np.argsort(off_diagonal, axis=None)[::-1], cm.shape))[0][:8]
    confusions = [f"- {symbols[t]} predicted as {symbols[p]}: {cm[t, p]} of {cm[t].sum()}"
                  for t, p in top if cm[t, p] > 0]

    text = "\n".join([
        "# Test set results",
        "",
        f"Test images: {len(y_test)}",
        f"**Test accuracy: {accuracy:.4f}** ({(y_pred == y_test).sum()} / {len(y_test)} correct)",
        "",
        "## Per-class report",
        "",
        "```",
        report,
        "```",
        "",
        "## Most common mistakes",
        "",
        *confusions,
        "",
    ])
    print(text)

    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "test_metrics.md").write_text(text)

    fig, ax = plt.subplots(figsize=(10, 10))
    ConfusionMatrixDisplay(cm, display_labels=symbols).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Test set confusion matrix (accuracy {accuracy:.2%})")
    fig.tight_layout()
    fig.savefig(DOCS_DIR / "confusion_matrix.png", dpi=120)
    print(f"Saved {DOCS_DIR / 'confusion_matrix.png'}")


if __name__ == "__main__":
    main()
