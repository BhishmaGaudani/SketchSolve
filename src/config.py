"""Shared settings: file paths and the random seed.

Everything that more than one script needs lives here, so there is a single
place to change it.
"""
from pathlib import Path

from tensorflow import keras

# Repo root = the folder that contains src/. Building paths from here means
# scripts work no matter which directory you run them from.
ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"            # the Kaggle dataset goes here (Phase 2)
PROCESSED_DATA_DIR = DATA_DIR / "processed"  # train/val/test splits (Phase 2)
MODELS_DIR = ROOT / "models"
DOCS_DIR = ROOT / "docs"

SEED = 42

# The symbols the model recognizes, as (symbol, dataset folders) pairs.
# The position in this list is the class index the model predicts.
# To add a class, append a line, e.g. ("(", ["("]), then rerun
# prepare_data.py and train.py.
CLASSES = [
    ("0", ["0"]), ("1", ["1"]), ("2", ["2"]), ("3", ["3"]), ("4", ["4"]),
    ("5", ["5"]), ("6", ["6"]), ("7", ["7"]), ("8", ["8"]), ("9", ["9"]),
    ("+", ["+"]),
    ("-", ["-"]),
    ("÷", ["div"]),
    ("=", ["="]),
    # One "x-shaped" class for both the variable x (dataset folder "X") and the
    # times sign ("times"). Handwritten, they look the same, and a model trained
    # on them separately couldn't tell them apart. src/solver.py decides which
    # one it is from the neighboring symbols.
    ("x", ["X", "times"]),
]

MODEL_PATH = MODELS_DIR / "symbol_cnn.keras"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"
SPLITS_PATH = PROCESSED_DATA_DIR / "splits.npz"


def set_seed(seed: int = SEED) -> None:
    """Seed Python, NumPy and TensorFlow so results are repeatable.

    This fixes the random weight init, shuffling and augmentation. Some
    TensorFlow ops can still add tiny run-to-run differences.
    """
    keras.utils.set_random_seed(seed)
