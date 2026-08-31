#!/usr/bin/env python3
"""Run a saved packing program and store the returned circles as CSV."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import math
from pathlib import Path


TOLERANCE = 1e-12


def load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("packing_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate(centers, radii, expected_count: int):
    if centers.shape != (expected_count, 2) or radii.shape != (expected_count,):
        raise RuntimeError(f"Expected {expected_count} circles, got centers={centers.shape}, radii={radii.shape}")
    if not all(math.isfinite(float(value)) for value in centers.flat):
        raise RuntimeError("A center coordinate is not finite")
    if not all(math.isfinite(float(value)) and float(value) >= 0 for value in radii):
        raise RuntimeError("A radius is negative or not finite")
    for index, (x, y) in enumerate(centers):
        radius = radii[index]
        if min(x - radius, y - radius, 1 - x - radius, 1 - y - radius) < -TOLERANCE:
            raise RuntimeError(f"Circle {index} crosses the square boundary")
    for first in range(expected_count):
        for second in range(first + 1, expected_count):
            dx = float(centers[first, 0] - centers[second, 0])
            dy = float(centers[first, 1] - centers[second, 1])
            distance = math.hypot(dx, dy)
            if distance < float(radii[first] + radii[second]) - TOLERANCE:
                raise RuntimeError(f"Circles {first} and {second} overlap")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("solution", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, required=True)
    args = parser.parse_args()

    solution = load_solution(args.solution)
    centers, radii, returned_sum = solution.run_packing()
    validate(centers, radii, args.count)
    calculated_sum = sum(float(radius) for radius in radii)
    if not math.isclose(calculated_sum, float(returned_sum), rel_tol=0, abs_tol=1e-10):
        raise RuntimeError(f"Returned sum {returned_sum!r} does not match calculated sum {calculated_sum!r}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["circle", "x", "y", "radius"])
        for index, (x, y) in enumerate(centers):
            radius = radii[index]
            writer.writerow([index, repr(float(x)), repr(float(y)), repr(float(radius))])

    print(f"circles={args.count} sum={calculated_sum:.16f} output={args.output}")


if __name__ == "__main__":
    main()
