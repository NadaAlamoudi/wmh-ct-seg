"""
Turn a predicted WMH mask into a volume in millilitres and a burden category.

Run: wmhct-wmh-volume --help

Note: The categories are volume-derived strata (10 and 25 mL), not Fazekas grades.
"""

import argparse
import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

LOW_ML = 10.0
HIGH_ML = 25.0


def burden_category(volume_ml, low=LOW_ML, high=HIGH_ML):
    """Volume-derived severity stratum. Not a Fazekas grade."""
    if not np.isfinite(volume_ml):
        return "unknown"
    if volume_ml <= low:
        return "low"
    if volume_ml <= high:
        return "medium"
    return "high"


def mask_volume_ml(path, label=None):
    """Foreground volume in mL, using the voxel size from the image header."""
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    voxel_mm3 = float(np.prod(img.header.get_zooms()[:3]))
    count = int((data == label).sum()) if label is not None else int((data > 0).sum())
    return count * voxel_mm3 / 1000.0, voxel_mm3, count


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--in_dir", type=Path, help="folder of predicted masks")
    src.add_argument("--in_file", type=Path, help="a single predicted mask")
    ap.add_argument("--out_csv", type=Path, default=None)
    ap.add_argument("--label", type=int, default=None,
                    help="count only this label value (default: any non-zero voxel)")
    ap.add_argument("--low_ml", type=float, default=LOW_ML)
    ap.add_argument("--high_ml", type=float, default=HIGH_ML)
    ap.add_argument("--pattern", default="*.nii.gz")
    args = ap.parse_args()

    if args.in_file:
        paths = [args.in_file]
    else:
        paths = sorted(p for p in args.in_dir.glob(args.pattern)
                       if p.name.endswith((".nii", ".nii.gz")))
    if not paths:
        raise SystemExit("No NIfTI files found. Check --in_dir and --pattern.")

    rows = []
    for path in paths:
        volume_ml, voxel_mm3, count = mask_volume_ml(path, args.label)
        rows.append({
            "scan": path.name.replace(".nii.gz", "").replace(".nii", ""),
            "wmh_volume_ml": volume_ml,
            "burden_category": burden_category(volume_ml, args.low_ml, args.high_ml),
            "voxel_volume_mm3": voxel_mm3,
            "foreground_voxels": count,
        })

    table = pd.DataFrame(rows)
    pd.set_option("display.width", 120)
    print(table[["scan", "wmh_volume_ml", "burden_category"]].to_string(
        index=False, float_format=lambda v: f"{v:.2f}"))

    counts = table["burden_category"].value_counts()
    print("\nBurden distribution: " +
          ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"Thresholds: low <= {args.low_ml:g} mL, medium <= {args.high_ml:g} mL, "
          f"high > {args.high_ml:g} mL. These are volume classes, not Fazekas grades.")

    empty = table[table["foreground_voxels"] == 0]
    if len(empty):
        print(f"\nWARNING: {len(empty)} mask(s) are empty. On non-contrast CT an empty "
              f"prediction can be genuine in a young or healthy brain, but check the "
              f"preprocessing first: {list(empty['scan'])[:5]}")

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
        table.to_csv(args.out_csv, index=False)
        print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
