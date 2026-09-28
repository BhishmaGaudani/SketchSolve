"""Load the train/val/test splits written by scripts/prepare_data.py."""
import numpy as np

from src.config import SPLITS_PATH


def load_split(name: str):
    """Return (x, y) for "train", "val" or "test".

    x has shape (n, 28, 28, 1) with float values in [0, 1]; y holds class indices.
    """
    if not SPLITS_PATH.exists():
        raise SystemExit(f"{SPLITS_PATH} not found. Run: python -m scripts.prepare_data")
    with np.load(SPLITS_PATH) as data:
        x = data[f"x_{name}"].astype(np.float32) / 255.0  # stored as uint8 0-255
        y = data[f"y_{name}"]
    return x[..., np.newaxis], y
