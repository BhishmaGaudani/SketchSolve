# End-to-end evaluation

1000 synthetic equations built from test-split images (seed 42), run through the full pipeline: segment -> classify -> solve. Headline numbers use a 9 px pen, which matches the app's pen width relative to symbol size.

| Metric | Value |
|---|---|
| Per-symbol accuracy (test split, single 28x28 images) | 98.83% |
| Per-symbol accuracy inside equations (correctly segmented ones) | 98.23% |
| **Full-equation accuracy (answer correct)** | **87.40%** (874/1000) |
| Expected if every symbol failed independently (mean of accuracy^length) | 93.53% |

## Sensitivity to pen thickness

Same equations and images, only the stroke width changes (digits are 120 px tall).

| Pen width | Full-equation accuracy | Segmentation failures | Classification failures |
|---|---|---|---|
| 5 px | 87.00% | 57 | 73 |
| 7 px | 87.20% | 45 | 83 |
| 9 px (headline) | 87.40% | 36 | 90 |
| 11 px | 85.50% | 18 | 127 |
| 13 px | 84.30% | 11 | 146 |

## Why equations fail (headline pen width)

| Cause | Equations | Share of all |
|---|---|---|
| Segmentation (wrong number of symbols) | 36 | 3.6% |
| Classification (a symbol misread) | 90 | 9.0% |
| Solver / context (symbols right, answer wrong) | 0 | 0.0% |

## Most common misreads inside equations (headline pen width)

Counted over correctly segmented equations; the x-shaped class covers both x and ×.

| True | Read as | Times |
|---|---|---|
| = | - | 16 |
| x | 1 | 10 |
| ÷ | 9 | 7 |
| 4 | 9 | 7 |
| x | 2 | 6 |
| 9 | 5 | 5 |
| 4 | 1 | 5 |
| 7 | + | 5 |

## Accuracy by expression length (headline pen width)

| Symbols | Equations | Correct |
|---|---|---|
| 3 | 23 | 91.3% |
| 4 | 154 | 88.3% |
| 5 | 412 | 92.0% |
| 6 | 108 | 86.1% |
| 7 | 151 | 80.1% |
| 8 | 101 | 78.2% |
| 9 | 51 | 88.2% |

## First failures (headline pen width)

Debug images for the first 8 are in `docs/e2e_failures/`.

| True | Read as | Answer given | Cause |
|---|---|---|---|
| 6x-17=49 | 6x-17-49 | There's an x but no '=', so there's nothing to solve for. | classification |
| 7x-3=-38 | 72-3=-38 | False (left side is 69, right side is -38) | classification |
| 7x+1=22 | 7x+1=x2 | x = -1/5 ≈ -0.2 | classification |
| 5x-10=-45 | 5-x-10=-45 | x = 40 | segmentation: found 10 symbols, expected 9 |
| 9x-9=45 | 922-9=45 | False (left side is 913, right side is 45) | segmentation: found 8 symbols, expected 7 |
| x+7=3 | 22+7=3 | False (left side is 29, right side is 3) | segmentation: found 6 symbols, expected 5 |
| 9÷9 | 5÷9 | 5/9 ≈ 0.555556 | classification |
| 38+46 | 38÷46 | 19/23 ≈ 0.826087 | classification |
| x+9=8 | x+9-8 | There's an x but no '=', so there's nothing to solve for. | classification |
| 18÷9 | 18÷5 | 18/5 ≈ 3.6 | classification |
| x+14=25 | 00+14=25 | False (left side is 14, right side is 25) | segmentation: found 8 symbols, expected 7 |
| x+16=7 | x+16-7 | There's an x but no '=', so there's nothing to solve for. | classification |
| 8x=-48 | 83x=-48 | x = -48/83 ≈ -0.578313 | segmentation: found 7 symbols, expected 6 |
| x+17=13 | x+77=13 | x = -64 | classification |
| x+12=23 | x+1x=23 | x = 23/2 ≈ 11.5 | classification |
| 4x-13=19 | 4x-13-19 | There's an x but no '=', so there's nothing to solve for. | classification |
| x+4=6 | 72+4=6 | False (left side is 76, right side is 6) | segmentation: found 6 symbols, expected 5 |
| x+19=23 | x+24=23 | x = -1 | classification |
| 51-41 | 52-41 | 11 | classification |
| 6x=-12 | 671=-12 | False (left side is 671, right side is -12) | segmentation: found 7 symbols, expected 6 |
| 4x+8=-24 | 41+8=-24 | False (left side is 49, right side is -24) | classification |
| 5x+5=45 | 5x+5-45 | There's an x but no '=', so there's nothing to solve for. | classification |
| 6x+3=-27 | 672+3=-27 | False (left side is 675, right side is -27) | segmentation: found 9 symbols, expected 8 |
| 5x=-30 | 51=-30 | False (left side is 51, right side is -30) | classification |
| 44÷4 | 4491 | 4491 | classification |
