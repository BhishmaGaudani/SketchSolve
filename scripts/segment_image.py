"""Run the full pipeline on a saved drawing, and save a debug image.

    python -m scripts.segment_image path/to/drawing.png

Works with dark-on-light or light-on-dark images. The debug image (boxes +
predicted symbols + the 28x28 model inputs) is saved to debug/.
"""
import sys
from pathlib import Path

import cv2

from src.config import ROOT
from src.pipeline import Recognizer


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    path = Path(sys.argv[1])
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"Could not read {path}")

    debug_path = ROOT / "debug" / f"{path.stem}_segments.png"
    result = Recognizer().recognize(gray, debug_path=debug_path)

    for i, sym in enumerate(result.symbols):
        flag = "  <- low confidence" if sym["low_confidence"] else ""
        print(f"{i}: {sym['symbol']}  confidence {sym['confidence']:.2f}{flag}")
    print("Expression:", result.expression)
    print("Answer:", result.answer if result.error is None else f"error: {result.error}")
    if result.symbols:
        print(f"Saved {debug_path}")


if __name__ == "__main__":
    main()
