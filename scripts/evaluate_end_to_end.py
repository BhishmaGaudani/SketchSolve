"""Measure per-symbol accuracy vs full-equation accuracy. Run from the repo root:

    python -m scripts.evaluate_end_to_end

Builds synthetic equations by placing TEST-split symbol images side by side
(scaled up with thick strokes, like a canvas drawing), runs the full app
pipeline (segment -> classify -> solve) and checks the answer.

Because pen thickness turned out to matter a lot (a thick pen closes the gap
between the two lines of "=" and joins the dots of "÷" to its line), the
same 1000 equations are run at several pen widths. The headline width
matches the app's pen relative to symbol size.

Writes docs/end_to_end.md and example failure images in docs/e2e_failures/.
"""
import random
from collections import Counter, defaultdict

import cv2
import numpy as np

from src.config import CLASSES, DOCS_DIR, RAW_DATA_DIR, SEED, SPLITS_PATH, set_seed
from src.pipeline import Recognizer
from src.preprocessing import load_dataset_image
from src.solver import solve_symbols

N_EQUATIONS = 1000
N_FAILURE_IMAGES = 8
DATASET_DIR = RAW_DATA_DIR / "extracted_images"

# Which dataset folder to draw each written symbol from. × and x share one
# class, so the folder decides whether we place a times sign or a variable.
SOURCE_FOLDER = {"×": "times", "x": "X", "÷": "div"}
# The class label the model should predict for each written symbol.
CLASS_OF = {"×": "x"}

TALL_SIZE, SMALL_SIZE = 120, 80                     # digits/x vs operators, in px
# The app's pen is 14 px on its canvas, where digits are drawn about 190 px
# tall: 14/190 of 120 px is about 9 px.
HEADLINE_STROKE = 9
STROKE_WIDTHS = [5, 7, 9, 11, 13]


# ---------- generating equations ----------

def make_equation(rng: random.Random) -> str:
    """A random arithmetic expression or linear equation with an integer answer."""
    if rng.random() < 0.5:
        op = rng.choice("+-×÷")
        if op == "÷":  # keep division exact so the answer is a whole number
            b, q = rng.randint(1, 12), rng.randint(1, 12)
            return f"{b * q}÷{b}"
        return f"{rng.randint(1, 99)}{op}{rng.randint(1, 99)}"

    x = rng.randint(-9, 12)
    a, b = rng.randint(2, 9), rng.randint(1, 20)
    form = rng.choice(["ax+b=c", "ax-b=c", "ax=c", "x+b=c"])
    if form == "ax+b=c":
        return f"{a}x+{b}={a * x + b}"
    if form == "ax-b=c":
        return f"{a}x-{b}={a * x - b}"
    if form == "ax=c":
        return f"{a}x={a * x}"
    return f"x+{b}={x + b}"


# ---------- drawing equations from test images ----------

def load_test_pools():
    """Group the test split's source files by written symbol."""
    with np.load(SPLITS_PATH) as data:
        labels, paths = data["y_test"], data["paths_test"]
    symbols = [symbol for symbol, _ in CLASSES]
    pools = defaultdict(list)
    for label, path in zip(labels, paths):
        folder = path.split("/")[0]
        written = {"times": "×", "X": "x"}.get(folder, symbols[label])
        pools[written].append(path)
    return pools


def render(text: str, pools, rng: random.Random, stroke_width: int) -> np.ndarray:
    """Place one random test image per symbol side by side on a black canvas."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (stroke_width, stroke_width))
    pieces = []
    for ch in text:
        img = load_dataset_image(DATASET_DIR / rng.choice(pools[ch]))
        size = TALL_SIZE if (ch.isdigit() or ch == "x") else SMALL_SIZE
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
        # The dataset's 1-px strokes become thick pen strokes, like the canvas.
        img = cv2.dilate(np.where(img > 60, 255, 0).astype(np.uint8), kernel)
        pieces.append(img)

    gaps = [rng.randint(15, 40) for _ in pieces]
    width = sum(p.shape[1] for p in pieces) + sum(gaps) + 40
    canvas = np.zeros((TALL_SIZE + 120, width), dtype=np.uint8)
    x = 20
    for piece, gap in zip(pieces, gaps):
        h, w = piece.shape
        y = 60 + (TALL_SIZE - h) // 2 + rng.randint(-10, 10)  # small baseline wobble
        canvas[y:y + h, x:x + w] = np.maximum(canvas[y:y + h, x:x + w], piece)
        x += w + gap
    return canvas


# ---------- evaluation ----------

def symbol_test_accuracy(recognizer) -> float:
    with np.load(SPLITS_PATH) as data:
        x = data["x_test"].astype(np.float32)[..., np.newaxis] / 255.0
        y = data["y_test"]
    predicted, _ = recognizer.classify(x)
    return float(np.mean(np.array(predicted) == np.array(recognizer.labels)[y]))


def run(recognizer, pools, per_symbol, stroke_width, failures_dir=None):
    """Evaluate N_EQUATIONS equations at one pen width. Returns a stats dict."""
    rng = random.Random(SEED)  # same equations and images for every pen width
    outcomes = Counter()
    by_length = defaultdict(lambda: [0, 0])    # length -> [correct, total]
    symbols_right = symbols_total = 0          # within correctly segmented equations
    misreads = Counter()                       # (true, predicted) inside equations
    predicted_independent = []                 # per-symbol accuracy ** length
    failure_rows = []

    for _ in range(N_EQUATIONS):
        text = make_equation(rng)
        true_labels = [CLASS_OF.get(ch, ch) for ch in text]
        truth = solve_symbols(true_labels)
        assert truth.error is None and truth.expression == text, text  # sanity check

        image = render(text, pools, rng, stroke_width)
        result = recognizer.recognize(image)
        predicted_labels = [CLASS_OF.get(s["symbol"], s["symbol"]) for s in result.symbols]

        correct = result.error is None and result.answer == truth.answer
        predicted_independent.append(per_symbol ** len(text))
        by_length[len(text)][1] += 1
        if len(predicted_labels) == len(true_labels):
            symbols_total += len(true_labels)
            symbols_right += sum(p == t for p, t in zip(predicted_labels, true_labels))
            misreads.update((t, p) for p, t in zip(predicted_labels, true_labels) if p != t)

        if correct:
            outcomes["correct"] += 1
            by_length[len(text)][0] += 1
            continue
        if len(predicted_labels) != len(true_labels):
            outcomes["segmentation"] += 1       # wrong number of symbols found
            reason = f"segmentation: found {len(predicted_labels)} symbols, expected {len(true_labels)}"
        elif predicted_labels != true_labels:
            outcomes["classification"] += 1    # right boxes, wrong symbol(s)
            reason = "classification"
        else:
            outcomes["solver"] += 1            # symbols right, answer still wrong
            reason = "solver"
        failure_rows.append(f"| {text} | {result.expression or '(nothing)'} | "
                            f"{result.answer or result.error} | {reason} |")
        if failures_dir is not None and len(failure_rows) <= N_FAILURE_IMAGES:
            recognizer.recognize(image, debug_path=failures_dir / f"failure_{len(failure_rows)}.png")

    return {
        "accuracy": outcomes["correct"] / N_EQUATIONS,
        "outcomes": outcomes,
        "by_length": by_length,
        "in_context": symbols_right / symbols_total if symbols_total else float("nan"),
        "expected": float(np.mean(predicted_independent)),
        "failure_rows": failure_rows,
        "misreads": misreads,
    }


def main():
    set_seed()
    recognizer = Recognizer()
    pools = load_test_pools()
    per_symbol = symbol_test_accuracy(recognizer)
    print(f"Per-symbol test accuracy: {per_symbol:.4f}")

    failures_dir = DOCS_DIR / "e2e_failures"
    failures_dir.mkdir(parents=True, exist_ok=True)
    for old in failures_dir.glob("*.png"):
        old.unlink()

    results = {}
    for width in STROKE_WIDTHS:
        results[width] = run(recognizer, pools, per_symbol, width,
                             failures_dir if width == HEADLINE_STROKE else None)
        print(f"  pen {width:2d} px: {results[width]['accuracy']:.2%}")

    n = N_EQUATIONS
    head = results[HEADLINE_STROKE]
    outcomes = head["outcomes"]
    lines = [
        "# End-to-end evaluation",
        "",
        f"{n} synthetic equations built from test-split images (seed {SEED}), "
        "run through the full pipeline: segment -> classify -> solve. "
        f"Headline numbers use a {HEADLINE_STROKE} px pen, which matches the app's "
        "pen width relative to symbol size.",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Per-symbol accuracy (test split, single 28x28 images) | {per_symbol:.2%} |",
        f"| Per-symbol accuracy inside equations (correctly segmented ones) | {head['in_context']:.2%} |",
        f"| **Full-equation accuracy (answer correct)** | **{head['accuracy']:.2%}** ({outcomes['correct']}/{n}) |",
        f"| Expected if every symbol failed independently (mean of accuracy^length) | {head['expected']:.2%} |",
        "",
        "## Sensitivity to pen thickness",
        "",
        "Same equations and images, only the stroke width changes (digits are 120 px tall).",
        "",
        "| Pen width | Full-equation accuracy | Segmentation failures | Classification failures |",
        "|---|---|---|---|",
        *[f"| {w} px{' (headline)' if w == HEADLINE_STROKE else ''} | {r['accuracy']:.2%} | "
          f"{r['outcomes']['segmentation']} | {r['outcomes']['classification']} |"
          for w, r in results.items()],
        "",
        "## Why equations fail (headline pen width)",
        "",
        "| Cause | Equations | Share of all |",
        "|---|---|---|",
        f"| Segmentation (wrong number of symbols) | {outcomes['segmentation']} | {outcomes['segmentation'] / n:.1%} |",
        f"| Classification (a symbol misread) | {outcomes['classification']} | {outcomes['classification'] / n:.1%} |",
        f"| Solver / context (symbols right, answer wrong) | {outcomes['solver']} | {outcomes['solver'] / n:.1%} |",
        "",
        "## Most common misreads inside equations (headline pen width)",
        "",
        "Counted over correctly segmented equations; the x-shaped class covers both x and ×.",
        "",
        "| True | Read as | Times |",
        "|---|---|---|",
        *[f"| {t} | {p} | {count} |" for (t, p), count in head["misreads"].most_common(8)],
        "",
        "## Accuracy by expression length (headline pen width)",
        "",
        "| Symbols | Equations | Correct |",
        "|---|---|---|",
        *[f"| {length} | {total} | {right / total:.1%} |"
          for length, (right, total) in sorted(head["by_length"].items())],
        "",
        "## First failures (headline pen width)",
        "",
        f"Debug images for the first {N_FAILURE_IMAGES} are in `docs/e2e_failures/`.",
        "",
        "| True | Read as | Answer given | Cause |",
        "|---|---|---|---|",
        *head["failure_rows"][:25],
        "",
    ]
    report = "\n".join(lines)
    print(report)
    (DOCS_DIR / "end_to_end.md").write_text(report)


if __name__ == "__main__":
    main()
