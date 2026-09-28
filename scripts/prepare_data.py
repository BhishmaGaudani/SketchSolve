"""Build train/val/test splits from the Kaggle "Handwritten Math Symbols" dataset.

Expects the class folders (see CLASSES in src/config.py) in
data/raw/extracted_images/. Run from the repo root:

    python -m scripts.prepare_data

Writes data/processed/splits.npz, docs/data_report.md and
docs/preprocessed_samples.png.
"""
import hashlib

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

from src.config import CLASSES, DOCS_DIR, RAW_DATA_DIR, SEED, SPLITS_PATH
from src.preprocessing import load_dataset_image, preprocess_symbol

DATASET_DIR = RAW_DATA_DIR / "extracted_images"
MAX_PER_CLASS = 4000        # cap big classes so no symbol dominates training
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15


def check_folders():
    missing = [folder for _, folders in CLASSES for folder in folders
               if not (DATASET_DIR / folder).is_dir()]
    if missing:
        raise SystemExit(f"Missing class folders in {DATASET_DIR}: {missing}")


def load_class(folders):
    """Preprocess every image in one class's folder(s).

    Returns the file count, a dict of {hash: image}, which drops exact
    duplicates (two files that become the same 28x28 image count once), and a
    dict of {hash: source file} so the end-to-end evaluation can reuse the
    original images.
    """
    images, sources = {}, {}
    files = [path for folder in folders for path in sorted((DATASET_DIR / folder).glob("*.jpg"))]
    for path in files:
        try:
            img = preprocess_symbol(load_dataset_image(path))
        except ValueError:
            continue  # blank image
        img = np.round(img[..., 0] * 255).astype(np.uint8)  # uint8 = 4x smaller file
        key = hashlib.md5(img.tobytes()).hexdigest()
        images[key] = img
        sources[key] = str(path.relative_to(DATASET_DIR))
    return len(files), images, sources


def save_sample_grid(x, y, path, per_class=10):
    """One row of example preprocessed images per class, for eyeballing."""
    rng = np.random.default_rng(SEED)
    rows = []
    for label in range(len(CLASSES)):
        idx = rng.choice(np.flatnonzero(y == label), size=per_class, replace=False)
        rows.append(np.hstack([np.pad(x[i], 1, constant_values=80) for i in idx]))
    grid = cv2.resize(np.vstack(rows), None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(str(path), grid)


def main():
    check_folders()
    rng = np.random.default_rng(SEED)

    print("Preprocessing images (takes a few minutes)...")
    raw_counts, per_class, per_class_sources = [], [], []
    for symbol, folders in CLASSES:
        n_files, images, sources = load_class(folders)
        raw_counts.append(n_files)
        per_class.append(images)
        per_class_sources.append(sources)
        print(f"  {symbol:>2}  {n_files:6d} files  {len(images):6d} unique")

    # An image that appears under two different labels would teach the model
    # contradictory answers, so drop it from every class it appears in.
    hash_sets = [set(images) for images in per_class]
    seen, conflicting = set(), set()
    for hashes in hash_sets:
        conflicting |= seen & hashes
        seen |= hashes

    xs, ys, paths, kept_counts = [], [], [], []
    for label, images in enumerate(per_class):
        keep = [h for h in images if h not in conflicting]
        if len(keep) > MAX_PER_CLASS:
            keep = list(rng.choice(keep, size=MAX_PER_CLASS, replace=False))
        xs.extend(images[h] for h in keep)
        paths.extend(per_class_sources[label][h] for h in keep)
        ys.extend([label] * len(keep))
        kept_counts.append(len(keep))

    x = np.stack(xs)
    y = np.array(ys, dtype=np.int64)
    paths = np.array(paths)

    # stratify=y keeps each class's share the same in every split. The paths
    # are split with the same shuffle, so paths_test[i] is the file of x_test[i].
    x_train, x_rest, y_train, y_rest, paths_train, paths_rest = train_test_split(
        x, y, paths, test_size=VAL_FRACTION + TEST_FRACTION, stratify=y, random_state=SEED)
    x_val, x_test, y_val, y_test, paths_val, paths_test = train_test_split(
        x_rest, y_rest, paths_rest, test_size=TEST_FRACTION / (VAL_FRACTION + TEST_FRACTION),
        stratify=y_rest, random_state=SEED)

    SPLITS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SPLITS_PATH, x_train=x_train, y_train=y_train,
                        x_val=x_val, y_val=y_val, x_test=x_test, y_test=y_test,
                        paths_train=paths_train, paths_val=paths_val, paths_test=paths_test)

    # Report, printed and saved for the README.
    lines = [
        "| Symbol | Files | Unique | Used | Train | Val | Test |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, (symbol, _) in enumerate(CLASSES):
        lines.append(
            f"| {symbol} | {raw_counts[label]} | {len(per_class[label])} | {kept_counts[label]} "
            f"| {(y_train == label).sum()} | {(y_val == label).sum()} | {(y_test == label).sum()} |")
    lines.append(
        f"| **Total** | {sum(raw_counts)} | {sum(len(p) for p in per_class)} | {len(y)} "
        f"| {len(y_train)} | {len(y_val)} | {len(y_test)} |")
    summary = [
        f"Images that appeared under two different labels (dropped): {len(conflicting)}",
        f"Classes capped at {MAX_PER_CLASS} images; split "
        f"{1 - VAL_FRACTION - TEST_FRACTION:.0%}/{VAL_FRACTION:.0%}/{TEST_FRACTION:.0%}, seed {SEED}.",
    ]
    report = "\n".join(["# Data report", "", *lines, "", *summary, ""])
    print("\n" + report)

    DOCS_DIR.mkdir(exist_ok=True)
    (DOCS_DIR / "data_report.md").write_text(report)
    save_sample_grid(x_train, y_train, DOCS_DIR / "preprocessed_samples.png")
    print(f"Saved splits to {SPLITS_PATH}")


if __name__ == "__main__":
    main()
