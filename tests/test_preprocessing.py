import cv2
import numpy as np
import pytest

from src.preprocessing import preprocess_symbol


def draw_plus(size, thickness, offset=(0, 0)):
    """A white "+" on black, drawn at a chosen size, stroke width and position."""
    img = np.zeros((300, 300), dtype=np.uint8)
    cx, cy = 150 + offset[0], 150 + offset[1]
    half = size // 2
    cv2.line(img, (cx - half, cy), (cx + half, cy), 255, thickness)
    cv2.line(img, (cx, cy - half), (cx, cy + half), 255, thickness)
    return img


def test_output_shape_and_range():
    out = preprocess_symbol(draw_plus(100, 10))
    assert out.shape == (28, 28, 1)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_blank_image_raises():
    with pytest.raises(ValueError):
        preprocess_symbol(np.zeros((50, 50), dtype=np.uint8))


def test_symbol_is_centered_by_mass():
    out = preprocess_symbol(draw_plus(60, 8, offset=(90, -80)))[..., 0]
    ys, xs = np.indices(out.shape)
    cx = (xs * out).sum() / out.sum()
    cy = (ys * out).sum() / out.sum()
    assert abs(cx - 14) < 1 and abs(cy - 14) < 1


def test_fits_in_20px_box():
    out = preprocess_symbol(draw_plus(250, 10))[..., 0]
    ys, xs = np.nonzero(out)
    assert xs.max() - xs.min() + 1 <= 21  # +1 px of slack for sub-pixel shift
    assert ys.max() - ys.min() + 1 <= 21


def test_thin_and_thick_strokes_give_similar_output():
    # The whole point of normalize_stroke: stroke width shouldn't matter.
    thin = preprocess_symbol(draw_plus(120, 2))
    thick = preprocess_symbol(draw_plus(120, 25))
    assert np.abs(thin - thick).mean() < 0.05


def test_divide_sign_keeps_both_dots():
    # Regression test: this exact drawing lost its bottom dot during thinning.
    img = np.zeros((117, 95), dtype=np.uint8)
    cv2.circle(img, (47, 8), 8, 255, -1)
    cv2.line(img, (7, 58), (87, 58), 255, 14)
    cv2.circle(img, (47, 108), 8, 255, -1)
    out = preprocess_symbol(img)[..., 0]
    n_pieces = cv2.connectedComponents((out > 0.3).astype(np.uint8))[0] - 1
    assert n_pieces == 3  # dot, line, dot


def test_aspect_ratio_is_kept_for_minus():
    img = np.zeros((100, 300), dtype=np.uint8)
    cv2.line(img, (20, 50), (280, 50), 255, 6)
    out = preprocess_symbol(img)[..., 0]
    ys, xs = np.nonzero(out > 0.5)
    assert (xs.max() - xs.min()) > 4 * (ys.max() - ys.min() + 1)  # still wide and flat
