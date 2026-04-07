"""Command line interface for fitting clmtrackr landmarks to an image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .tracker import CLMTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit the clmtrackr facial model to a still image.")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("-o", "--output", help="Optional output image with landmarks drawn")
    parser.add_argument("--json", dest="json_output", help="Optional JSON file for landmark coordinates")
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=int,
        metavar=("X", "Y", "W", "H"),
        help="Optional face box if automatic detection misses the face",
    )
    parser.add_argument("--max-iterations", type=int, default=40)
    parser.add_argument(
        "--min-iterations",
        type=int,
        default=10,
        help="Minimum fitting iterations before convergence can stop the optimizer",
    )
    args = parser.parse_args()

    tracker = CLMTracker()
    result = tracker.fit(
        args.image,
        bbox=tuple(args.bbox) if args.bbox else None,
        max_iterations=args.max_iterations,
        min_iterations=args.min_iterations,
    )

    if args.output:
        drawn = tracker.draw(args.image, result)
        cv2.imwrite(args.output, drawn)

    payload = {
        "bbox": result.bbox,
        "iterations": result.iterations,
        "converged": result.converged,
        "movement": result.movement,
        "points": result.points.tolist(),
    }
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
