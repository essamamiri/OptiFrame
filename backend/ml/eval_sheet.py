"""Grade the live measurement pipeline against caliper ground truth.

Runs the exact same calibration + measurement code the app uses on every
photo listed in ml/data/sheet/manifest.csv and compares A/B with the
caliper numbers. No masks or training needed, just photos + manifest.

Usage (from backend/):
    python -m ml.eval_sheet
    python -m ml.eval_sheet --data path/to/folder   # folder with manifest.csv + photos/
"""
from __future__ import annotations

import argparse
import pathlib
from collections import defaultdict

import numpy as np
from PIL import Image, ImageOps

from app.main import calibrate_with_supported_references
from app.measurement import measure_lens
from .sheet_data import DATA_DIR, load_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=pathlib.Path, default=DATA_DIR)
    args = ap.parse_args()

    records = load_manifest(args.data / "manifest.csv")
    if not records:
        print(f"No rows in {args.data / 'manifest.csv'}. See ml/data/sheet/README.md.")
        return

    errors: dict[str, list[tuple[float, float]]] = defaultdict(list)
    failed = 0
    print(f"{'photo':<22}{'lens':<10}{'A err':>8}{'B err':>8}  status")
    for rec in records:
        path = args.data / "photos" / rec["filename"]
        if not path.exists():
            print(f"{rec['filename']:<22}{rec['lens_id']:<10}{'':>16}  missing file")
            failed += 1
            continue
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        calibration = calibrate_with_supported_references(image)
        if calibration.status != "calibrated":
            print(f"{rec['filename']:<22}{rec['lens_id']:<10}{'':>16}  calibration: {calibration.reason}")
            failed += 1
            continue
        result = measure_lens(image, calibration, "loose")
        if result.status != "measured":
            print(f"{rec['filename']:<22}{rec['lens_id']:<10}{'':>16}  measure: {result.reason}")
            failed += 1
            continue
        err_a = result.width_a_mm - rec["width_a_mm"]
        err_b = result.height_b_mm - rec["height_b_mm"]
        errors[rec["lens_id"]].append((err_a, err_b))
        print(f"{rec['filename']:<22}{rec['lens_id']:<10}{err_a:>+8.2f}{err_b:>+8.2f}  ok")

    print()
    all_errs = np.array([e for errs in errors.values() for e in errs]) if errors else np.empty((0, 2))
    for lens_id, errs in sorted(errors.items()):
        e = np.array(errs)
        print(f"{lens_id}: n={len(e)}  mean|A|={np.abs(e[:, 0]).mean():.2f} mm  "
              f"mean|B|={np.abs(e[:, 1]).mean():.2f} mm  bias A={e[:, 0].mean():+.2f}  B={e[:, 1].mean():+.2f}")
    if len(all_errs):
        print(f"ALL: n={len(all_errs)}  mean|A|={np.abs(all_errs[:, 0]).mean():.2f} mm  "
              f"mean|B|={np.abs(all_errs[:, 1]).mean():.2f} mm  "
              f"worst={np.abs(all_errs).max():.2f} mm  failed={failed}/{len(records)}")
        bias = all_errs.mean(axis=0)
        spread = all_errs.std(axis=0)
        print(f"Systematic bias A={bias[0]:+.2f} B={bias[1]:+.2f} mm, scatter A={spread[0]:.2f} B={spread[1]:.2f} mm.")
        print("If the bias is much bigger than the scatter, it's a constant offset you can subtract.")


if __name__ == "__main__":
    main()
