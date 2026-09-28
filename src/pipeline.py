"""The full SketchSolve pipeline: drawing in, recognized expression and answer out.

    image -> segment into symbols -> classify each -> solve

Used by the Flask app and by the end-to-end evaluation, so both run exactly
the same code.
"""
import json
from dataclasses import dataclass, field

import numpy as np
from tensorflow import keras

from src.config import LABEL_MAP_PATH, MODEL_PATH
from src.segmentation import save_debug_image, segment
from src.solver import resolve_x, solve_symbols

# Symbols the model is less sure about than this are flagged in the UI.
# It's a rule of thumb, not a calibrated probability: softmax scores are
# often overconfident, so a wrong symbol can still score 0.9+.
LOW_CONFIDENCE = 0.8

EMPTY_MESSAGE = "The canvas is empty. Draw an expression first."


def ink_on_black(gray: np.ndarray) -> np.ndarray:
    """Return the drawing as white ink on black, flipping it if needed.

    Most of a drawing is background, so a mostly-light image is dark ink on
    a light background.
    """
    return 255 - gray if np.median(gray) > 127 else gray


@dataclass
class Recognition:
    symbols: list[dict] = field(default_factory=list)
    expression: str = ""
    parsed: str = ""
    answer: str | None = None
    error: str | None = None


class Recognizer:
    """Loads the model once and runs the pipeline on drawings."""

    def __init__(self, model_path=MODEL_PATH, label_map_path=LABEL_MAP_PATH):
        self.model = keras.models.load_model(model_path)
        label_map = json.loads(label_map_path.read_text())
        self.labels = [label_map[str(i)] for i in range(len(label_map))]

    def classify(self, inputs: np.ndarray) -> tuple[list[str], np.ndarray]:
        """Predict a label for each (28, 28, 1) input. Returns labels and confidences."""
        # Calling the model directly is faster than model.predict() for one
        # small batch, and training=False keeps augmentation/Dropout off.
        probs = self.model(inputs, training=False).numpy()
        return [self.labels[i] for i in probs.argmax(axis=1)], probs.max(axis=1)

    def recognize(self, gray: np.ndarray, debug_path=None) -> Recognition:
        """Run the whole pipeline on a grayscale drawing (any ink color)."""
        image = ink_on_black(gray)
        symbols = segment(image)
        if not symbols:
            return Recognition(error=EMPTY_MESSAGE)

        labels, confidences = self.classify(np.stack([s.input for s in symbols]))
        resolved = resolve_x(labels)  # × or x, as the solver will read them
        solved = solve_symbols(labels)

        if debug_path is not None:
            save_debug_image(image, symbols, debug_path, labels=resolved)

        return Recognition(
            symbols=[
                {
                    "symbol": symbol,
                    "confidence": round(float(conf), 4),
                    "low_confidence": bool(conf < LOW_CONFIDENCE),
                    "box": [s.box.x, s.box.y, s.box.w, s.box.h],
                }
                for s, symbol, conf in zip(symbols, resolved, confidences)
            ],
            expression=solved.expression,
            parsed=solved.parsed,
            answer=solved.answer,
            error=solved.error,
        )
