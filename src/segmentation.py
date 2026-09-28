"""Split a drawing of an expression into one crop per symbol, left to right.

Steps:
  1. threshold the image to pure black/white
  2. find each connected blob of ink (cv2.findContours)
  3. merge blobs that sit above each other (the two lines of "=", the dots
     and line of "÷", a "5" whose top bar doesn't touch its body)
  4. drop leftover specks that are too small to be a symbol
  5. run each symbol through the same preprocessing used in training
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.preprocessing import preprocess_symbol

# Merge two boxes when their x-ranges overlap by more than this fraction of
# the narrower box. The dots of "÷" lie fully inside the line's x-range
# (overlap 1.0); two neighboring digits usually overlap little or not at all.
MERGE_OVERLAP = 0.5
# Contours with a smaller box area than this (in pixels) are noise.
MIN_CONTOUR_AREA = 20
# After merging, a group whose longest side is smaller than this fraction of
# the biggest group's longest side is dropped (e.g. an accidental dot).
MIN_RELATIVE_SIZE = 0.25


@dataclass
class Box:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self):
        return self.x + self.w

    @property
    def y2(self):
        return self.y + self.h

    def union(self, other: "Box") -> "Box":
        x, y = min(self.x, other.x), min(self.y, other.y)
        return Box(x, y, max(self.x2, other.x2) - x, max(self.y2, other.y2) - y)


@dataclass
class Symbol:
    box: Box            # position in the original drawing
    crop: np.ndarray    # the symbol's own ink only, white on black
    input: np.ndarray   # (28, 28, 1) model input from preprocess_symbol


def x_overlap_ratio(a: Box, b: Box) -> float:
    """How much two boxes overlap horizontally, as a fraction of the narrower one."""
    overlap = min(a.x2, b.x2) - max(a.x, b.x)
    return max(0, overlap) / min(a.w, b.w)


def group_boxes(boxes: list[Box]) -> list[list[int]]:
    """Group box indices whose x-ranges overlap heavily. Returns groups left to right.

    Boxes are visited left to right; each one joins the current group if it
    overlaps that group's combined box enough, otherwise it starts a new group.
    """
    order = sorted(range(len(boxes)), key=lambda i: boxes[i].x)
    groups, group_box = [], None
    for i in order:
        if group_box is not None and x_overlap_ratio(group_box, boxes[i]) > MERGE_OVERLAP:
            groups[-1].append(i)
            group_box = group_box.union(boxes[i])
        else:
            groups.append([i])
            group_box = boxes[i]
    return groups


def union_all(boxes: list[Box]) -> Box:
    result = boxes[0]
    for box in boxes[1:]:
        result = result.union(box)
    return result


def segment(image: np.ndarray, debug_path=None) -> list[Symbol]:
    """Find the symbols in a drawing.

    Args:
        image: grayscale uint8 drawing, white ink on a black background.
        debug_path: if given, save an image there showing the boxes found and
            the 28x28 input made for each symbol.

    Returns:
        Symbols sorted left to right. Empty list if there's no ink.
    """
    binary = np.where(image > 127, 255, 0).astype(np.uint8)
    # RETR_EXTERNAL: only outer outlines, so the hole in "0" isn't a separate contour.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [Box(*cv2.boundingRect(c)) for c in contours]
    keep = [i for i, b in enumerate(boxes) if b.w * b.h >= MIN_CONTOUR_AREA]
    contours = [contours[i] for i in keep]
    boxes = [boxes[i] for i in keep]
    if not boxes:
        return []

    groups = group_boxes(boxes)
    merged = [union_all([boxes[i] for i in g]) for g in groups]
    biggest = max(max(b.w, b.h) for b in merged)

    symbols = []
    for group, box in zip(groups, merged):
        if max(box.w, box.h) < MIN_RELATIVE_SIZE * biggest:
            continue
        # Take only this group's own ink: fill its contours into a mask and
        # AND it with the drawing. A plain rectangle crop could include bits
        # of a neighboring symbol that reach into the box.
        mask = np.zeros_like(binary)
        cv2.drawContours(mask, [contours[i] for i in group], -1, 255, thickness=cv2.FILLED)
        ink = cv2.bitwise_and(binary, mask)
        crop = ink[box.y:box.y2, box.x:box.x2]
        symbols.append(Symbol(box=box, crop=crop, input=preprocess_symbol(crop)))

    if debug_path is not None:
        save_debug_image(image, symbols, debug_path)
    return symbols


def save_debug_image(image, symbols, path, labels=None):
    """Save the drawing with numbered boxes, plus each symbol's 28x28 input below it.

    `labels` (optional) are text labels, e.g. predictions, drawn instead of numbers.
    """
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for i, sym in enumerate(symbols):
        b = sym.box
        cv2.rectangle(canvas, (b.x, b.y), (b.x2, b.y2), (0, 200, 255), 2)
        text = labels[i] if labels is not None else str(i)
        cv2.putText(canvas, text, (b.x, max(b.y - 6, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 200, 255), 2)

    # Strip of model inputs, scaled up 3x, under the drawing.
    tile = 84
    strip = np.zeros((tile + 10, canvas.shape[1], 3), dtype=np.uint8)
    for i, sym in enumerate(symbols):
        x0 = 5 + i * (tile + 5)
        if x0 + tile > strip.shape[1]:
            break  # more symbols than fit across; the boxes above still show them
        small = cv2.resize((sym.input[..., 0] * 255).astype(np.uint8), (tile, tile),
                           interpolation=cv2.INTER_NEAREST)
        strip[5:5 + tile, x0:x0 + tile] = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack([canvas, strip]))
