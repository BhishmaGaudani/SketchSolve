"""Turn one symbol image into the 28x28 array the model expects.

The SAME function is used for the training data (scripts/prepare_data.py) and
for crops from the user's drawing (src/segmentation.py). If the two paths
preprocessed differently, the model would see inputs at prediction time that
look unlike anything it trained on.

Steps, following how MNIST was built:
  1. binarize and crop to the ink
  2. normalize stroke thickness (thin to a 1-pixel skeleton, then thicken)
  3. scale to fit a 20x20 box, keeping the aspect ratio
  4. paste into a 28x28 image and shift so the center of mass is in the middle
"""
import cv2
import numpy as np

OUTPUT_SIZE = 28
BOX_SIZE = 20       # the symbol fits inside this box; the rest is padding
WORK_SIZE = 64      # resolution used for the stroke-thickness step
STROKE_KERNEL = 7   # stroke width at WORK_SIZE; ~2 px after scaling to 20


def load_dataset_image(path) -> np.ndarray:
    """Read a dataset image and flip it to white ink on black, like the canvas."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    return 255 - img  # dataset images are black ink on white


def crop_to_ink(binary: np.ndarray) -> np.ndarray:
    """Cut away the empty rows/columns around the ink."""
    ys, xs = np.nonzero(binary)
    return binary[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def resize_to_fit(img: np.ndarray, size: int) -> np.ndarray:
    """Scale so the longest side equals `size`, keeping the aspect ratio."""
    h, w = img.shape
    scale = size / max(h, w)
    new_w = max(1, round(w * scale))  # max(1, ...) keeps thin lines like "-" from vanishing
    new_h = max(1, round(h * scale))
    # INTER_AREA averages pixels when shrinking (smooth edges);
    # INTER_LINEAR is the better choice when enlarging.
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def normalize_stroke(binary: np.ndarray) -> np.ndarray:
    """Make every stroke the same thickness, however it was drawn.

    Dataset strokes are ~1 px thin; canvas strokes are thick. Thinning both
    to a 1-pixel skeleton and thickening by a fixed amount makes them match.
    """
    work = resize_to_fit(binary, WORK_SIZE)
    work = np.where(work > 64, 255, 0).astype(np.uint8)
    # Padding stops ink touching the border, which confuses thinning and
    # would clip the thickened stroke.
    pad = STROKE_KERNEL
    work = cv2.copyMakeBorder(work, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)

    skeleton = cv2.ximgproc.thinning(work)
    # Thinning can erase a small blob completely (it happened to one dot of
    # a "÷" but not the other). Put back the center of any piece of ink that
    # lost all its pixels, so dots survive.
    n, labels, _, centroids = cv2.connectedComponentsWithStats(work)
    for i in range(1, n):  # label 0 is the background
        if not skeleton[labels == i].any():
            cx, cy = np.round(centroids[i]).astype(int)
            skeleton[cy, cx] = 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (STROKE_KERNEL, STROKE_KERNEL))
    return cv2.dilate(skeleton, kernel)


def center_by_mass(img: np.ndarray) -> np.ndarray:
    """Shift the image so its center of mass sits at the image center."""
    moments = cv2.moments(img)
    if moments["m00"] == 0:
        return img
    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]
    center = OUTPUT_SIZE / 2
    shift = np.float32([[1, 0, center - cx], [0, 1, center - cy]])
    return cv2.warpAffine(img, shift, (OUTPUT_SIZE, OUTPUT_SIZE), borderValue=0)


def preprocess_symbol(img: np.ndarray) -> np.ndarray:
    """Convert one symbol image to model input.

    Args:
        img: grayscale uint8 image of a single symbol, white ink on a black
            background, any size.

    Returns:
        float32 array of shape (28, 28, 1) with values in [0, 1].

    Raises:
        ValueError: if the image contains no ink.
    """
    binary = np.where(img > 127, 255, 0).astype(np.uint8)
    if not binary.any():
        raise ValueError("Image contains no ink")

    symbol = crop_to_ink(binary)
    symbol = normalize_stroke(symbol)
    symbol = crop_to_ink(symbol)  # thickening changed the ink's extent
    symbol = resize_to_fit(symbol, BOX_SIZE)  # grayscale edges, like MNIST

    # Paste into the middle of a blank 28x28 canvas, then fine-tune by mass.
    out = np.zeros((OUTPUT_SIZE, OUTPUT_SIZE), dtype=np.uint8)
    h, w = symbol.shape
    top = (OUTPUT_SIZE - h) // 2
    left = (OUTPUT_SIZE - w) // 2
    out[top:top + h, left:left + w] = symbol
    out = center_by_mass(out)

    return (out.astype(np.float32) / 255.0)[..., np.newaxis]
