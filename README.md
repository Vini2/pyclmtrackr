# pyclmtrackr

Python-only facial landmark fitting on still images using the original [clmtrackr](https://github.com/auduno/clmtrackr) model.

This repository has been cleaned down to the Python implementation. The browser JavaScript library, HTML demos, generated builds, and Node tooling have been removed. The default model is now packaged as JSON at `pyclmtrackr/data/model_pca_20_svm.json`.

## What It Does

`pyclmtrackr` fits a 71-point clmtrackr facial model to still images:

- Detects an initial frontal face box with OpenCV.
- Refines initialization using detected eyes when possible.
- Runs the CLM/SVM patch-response fitting loop in Python.
- Returns landmarks as a NumPy array shaped `(71, 2)`.
- Can write JSON landmarks and an overlay image from the CLI.

## Install

```bash
python3 -m pip install -r requirements.txt
```

If your default `python3` does not include `pip` or OpenCV, use the Python environment where `numpy` and `opencv-python` are installed.

## Command Line

```bash
python3 -m pyclmtrackr examples/franck_02159.jpg \
  --output /tmp/franck_landmarks.jpg \
  --json /tmp/franck_landmarks.json
```

If automatic detection misses the face, pass a manual face box:

```bash
python3 -m pyclmtrackr face.jpg --bbox 120 80 240 240 --json landmarks.json
```

The box format is:

```text
x y width height
```

## Python API

```python
import cv2
from pyclmtrackr import CLMTracker

tracker = CLMTracker()
result = tracker.fit("examples/franck_02159.jpg")

print(result.points.shape)  # (71, 2)
print(result.points[62])    # Nose landmark

overlay = tracker.draw("examples/franck_02159.jpg", result)
cv2.imwrite("/tmp/franck_landmarks.jpg", overlay)
```

Manual face box:

```python
result = tracker.fit("face.jpg", bbox=(120, 80, 240, 240))
```

For visual comparisons, make sure you run the fitter on the exact same image dimensions as the reference overlay. The original clmtrackr demo images are often cropped/resized, so overlay coordinates from `examples/franck_02159.jpg` will not line up pixel-for-pixel with the old `clmtrackr_03.jpg` README figure.

## Project Layout

```text
.
├── LICENSE.txt
├── README.md
├── pyproject.toml
├── requirements.txt
├── examples/
│   └── franck_02159.jpg
└── pyclmtrackr/
    ├── __init__.py
    ├── __main__.py
    ├── model.py
    ├── tracker.py
    └── data/
        └── model_pca_20_svm.json
```

## Notes

This is intentionally not a full port of the original browser/video tracker. It focuses on fitting the facial model to still images in Python.

The original clmtrackr project was distributed under the MIT License; keep `LICENSE.txt` with redistributed copies.
