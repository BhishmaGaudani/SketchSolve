import cv2
import numpy as np
import pytest

from src.segmentation import Box, group_boxes, segment, x_overlap_ratio

STROKE = 12


# ---- merge logic on plain boxes ----

def test_overlap_ratio_uses_narrower_box():
    wide = Box(0, 0, 100, 10)
    narrow = Box(40, 20, 10, 10)  # fully inside wide's x-range
    assert x_overlap_ratio(wide, narrow) == 1.0


def test_no_overlap_is_zero():
    assert x_overlap_ratio(Box(0, 0, 10, 10), Box(50, 0, 10, 10)) == 0


def test_equals_sign_lines_are_merged():
    top, bottom = Box(10, 40, 80, 8), Box(12, 70, 78, 8)
    assert group_boxes([top, bottom]) == [[0, 1]]


def test_divide_sign_parts_are_merged():
    dot1, line, dot2 = Box(45, 10, 10, 10), Box(10, 40, 80, 8), Box(46, 70, 10, 10)
    assert len(group_boxes([dot1, line, dot2])) == 1


def test_neighboring_digits_stay_separate():
    assert group_boxes([Box(0, 0, 50, 100), Box(60, 0, 50, 100)]) == [[0], [1]]


def test_slight_overlap_stays_separate():
    # Slanted handwriting: boxes overlap by 20% of their width.
    assert len(group_boxes([Box(0, 0, 50, 100), Box(40, 0, 50, 100)])) == 2


def test_groups_come_out_left_to_right():
    assert group_boxes([Box(200, 0, 50, 50), Box(0, 0, 50, 50)]) == [[1], [0]]


# ---- full segment() on drawn images ----

def blank(w=600, h=200):
    return np.zeros((h, w), dtype=np.uint8)


def test_empty_drawing_gives_no_symbols():
    assert segment(blank()) == []


def test_one_equals_one():
    img = blank()
    cv2.line(img, (60, 40), (60, 160), 255, STROKE)       # 1
    cv2.line(img, (200, 80), (320, 80), 255, STROKE)      # = top
    cv2.line(img, (200, 120), (320, 120), 255, STROKE)    # = bottom
    cv2.line(img, (450, 40), (450, 160), 255, STROKE)     # 1
    symbols = segment(img)
    assert len(symbols) == 3
    middle = symbols[1].box
    assert middle.y < 80 and middle.y2 > 120  # one box around both lines


def test_divide_sign_is_one_symbol():
    img = blank()
    cv2.circle(img, (300, 50), 8, 255, -1)
    cv2.line(img, (240, 100), (360, 100), 255, STROKE)
    cv2.circle(img, (300, 150), 8, 255, -1)
    assert len(segment(img)) == 1


def test_zero_with_hole_is_one_symbol():
    img = blank()
    cv2.ellipse(img, (300, 100), (40, 70), 0, 0, 360, 255, STROKE)
    assert len(segment(img)) == 1


def test_tiny_specks_are_ignored():
    img = blank()
    cv2.line(img, (100, 40), (100, 160), 255, STROKE)
    img[10:12, 500:502] = 255       # 2x2 noise, removed by MIN_CONTOUR_AREA
    cv2.circle(img, (400, 100), 5, 255, -1)  # lone dot, removed by MIN_RELATIVE_SIZE
    assert len(segment(img)) == 1


def test_crop_contains_only_its_own_ink():
    # Left symbol: an "L" whose tail reaches into the right symbol's box.
    img = blank()
    cv2.line(img, (100, 40), (100, 160), 255, STROKE)
    cv2.line(img, (100, 160), (230, 160), 255, STROKE)   # tail ends at x=230
    # Right symbol: a "7" whose box spans x 194-306, y 34-176.
    cv2.line(img, (200, 40), (300, 40), 255, STROKE)
    cv2.line(img, (300, 40), (300, 170), 255, STROKE)
    symbols = segment(img)
    assert len(symbols) == 2

    right = symbols[1]
    assert right.box.x < 230 and right.box.y2 > 160  # the tail IS inside its rectangle
    tail_y = 160 - right.box.y
    tail_x_end = 230 + STROKE - right.box.x
    assert right.crop[tail_y - 6:tail_y + 6, :tail_x_end].max() == 0  # ...but not in its crop


@pytest.mark.xfail(reason="Known limitation: a symbol drawn as separate strokes side by "
                          "side (like an open '4') is split in two. See README limitations.")
def test_two_stroke_four_is_one_symbol():
    img = blank()
    cv2.line(img, (100, 40), (80, 120), 255, STROKE)     # left stroke of "4"
    cv2.line(img, (80, 120), (150, 120), 255, STROKE)    # crossbar
    cv2.line(img, (165, 60), (165, 170), 255, STROKE)    # separate right stroke
    assert len(segment(img)) == 1
