"""
Stage 6.1 - Compute WMH volumes in CT space and native FLAIR space for every scan.

Manuscript: source table for Figures 10 and 11 and the volume agreement analysis
Run: wmhct-volumes --help
"""

import argparse
import os
import re

import nibabel as nib
import numpy as np
import pandas as pd

from wmhct import config


def scan_id_from_name(fname):
    """'MSS3_9011.nii.gz' -> '0281'; '0281_gt.nii.gz' -> '0281'."""
    m = re.match(r'^(?:MSS3_)?(\d+)', fname)
    return m.group(1) if m else None


def index_by_id(folder):
    """Map exact scan ID -> path. Raises on duplicate or unparseable IDs."""
    idx = {}
    for f in sorted(os.listdir(folder)):
        if not f.endswith(('.nii', '.nii.gz')):
            continue
        sid = scan_id_from_name(f)
        if sid is None:
            raise ValueError(f"cannot parse scan ID from {f}")
        if sid in idx:
            raise ValueError(f"duplicate ID {sid} in {folder}: "
                             f"{os.path.basename(idx[sid])} and {f}")
        idx[sid] = os.path.join(folder, f)
    return idx


def compute_volume(mask_img, wmh_label=None):
    """Volume in mL. If wmh_label is given, count only that label."""
    data = mask_img.get_fdata()
    vox_mm3 = float(np.prod(mask_img.header.get_zooms()[:3]))
    n = (data == wmh_label).sum() if wmh_label is not None else (data > 0).sum()
    return n * vox_mm3 / 1000.0


def main():
    parser = argparse.ArgumentParser(description="Compute WMH volumes in CT and FLAIR space")
    parser.add_argument("--ct_pred_dir", default=str(config.MODEL7_PREDICTIONS_CT_SPACE))
    parser.add_argument("--ct_gt_dir", required=True,
                        help="Reference WMH masks resampled into CT space")
    parser.add_argument("--flair_gt_dir", required=True,
                        help="Reference WMH masks in native FLAIR space")
    parser.add_argument("--flair_pred_dir", default=str(config.MODEL7_PREDICTIONS_FLAIR_SPACE))
    parser.add_argument("--out_csv", default=str(config.VOLUME_TABLE))
    args = parser.parse_args()

    folders = {
        "CT nnUNet": args.ct_pred_dir,
        "Registered GT": args.ct_gt_dir,
        "FLAIR GT": args.flair_gt_dir,
        "FLAIR nnUNet": args.flair_pred_dir,
    }

    for name, folder in folders.items():
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"{name} folder not found: {folder}")
        n = len([f for f in os.listdir(folder) if f.endswith(('.nii', '.nii.gz'))])
        print(f"{name}: {n} NIfTI files")

    ct_idx = index_by_id(args.ct_pred_dir)
    reg_gt_idx = index_by_id(args.ct_gt_dir)
    fl_gt_idx = index_by_id(args.flair_gt_dir)
    fl_nn_idx = index_by_id(args.flair_pred_dir)

    all_ids = set(ct_idx)
    for nm, idx in [("regGT", reg_gt_idx), ("flGT", fl_gt_idx), ("flNN", fl_nn_idx)]:
        if set(idx) != all_ids:
            raise ValueError(f"{nm} ID mismatch. "
                             f"missing={sorted(all_ids - set(idx))} "
                             f"extra={sorted(set(idx) - all_ids)}")

    rows = []
    for sid in sorted(all_ids):
        rows.append({
            'Patient ID': sid,  # kept as a string so leading zeros survive
            'nnUnet CT-based Volume (ml)': compute_volume(nib.load(ct_idx[sid])),
            'Registered to CT_space GT Volume (ml)': compute_volume(nib.load(reg_gt_idx[sid])),
            'FLAIR GT Volume (ml)': compute_volume(nib.load(fl_gt_idx[sid])),
            'nnUNet Volume in FLAIR_space (ml)': compute_volume(nib.load(fl_nn_idx[sid])),
        })

    results_df = pd.DataFrame(rows)
    assert results_df.notna().all().all(), "NaNs present - check the folder indices"

    out_dir = os.path.dirname(os.path.abspath(args.out_csv))
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(args.out_csv, index=False)

    # Sanity check reported in the revision: CT-space and FLAIR-space volumes are the
    # same mask in two spaces, so their ratio should be tight. A wide spread indicates
    # a file-matching fault.
    ratio = (results_df['nnUnet CT-based Volume (ml)']
             / results_df['nnUNet Volume in FLAIR_space (ml)'].replace(0, np.nan))
    print(f"\nWrote {len(results_df)} rows to {args.out_csv}")
    print(f"CT/FLAIR predicted-volume ratio: mean {ratio.mean():.4f}, "
          f"SD {ratio.std(ddof=1):.4f}")
    if ratio.std(ddof=1) > 0.05:
        print("WARNING: the ratio spread is large. In correct data the SD is under 0.01. "
              "Check for duplicate or mismatched files.")


if __name__ == "__main__":
    main()
