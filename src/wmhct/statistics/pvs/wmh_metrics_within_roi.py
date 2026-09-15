"""
Supporting analysis - WMH segmentation performance inside the PVS ROIs.

Manuscript: PVS ROI analysis
Run: wmhct-wmh-metrics-by-roi --help
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


def load_bool(path) -> np.ndarray:
    return nib.load(str(path)).get_fdata() > 0


def confusion(gt, pr, domain):
    g = gt[domain]
    p = pr[domain]
    return {
        "TP": int(np.sum(g & p)),
        "FP": int(np.sum((~g) & p)),
        "FN": int(np.sum(g & (~p))),
        "TN": int(np.sum((~g) & (~p))),
    }


def safe_rate(num, den):
    return float(num / den) if den != 0 else np.nan


def derive_metrics(c):
    tp, fp, fn, tn = c["TP"], c["FP"], c["FN"], c["TN"]
    out = dict(c)
    out["TPF"] = safe_rate(tp, tp + fn)
    out["FPR"] = safe_rate(fp, fp + tn)
    out["FDR"] = safe_rate(fp, tp + fp)
    out["Dice"] = safe_rate(2 * tp, 2 * tp + fp + fn)
    return out


def pack(prefix, d):
    return {f"{prefix}_{k}": v for k, v in d.items()}


def main():
    parser = argparse.ArgumentParser(description="WMH metrics within the PVS ROIs")
    parser.add_argument("--manifest", required=True,
                        help="roi_wmh_manifest_V1.csv, prepared upstream")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--id_col", default="subject_id")
    parser.add_argument("--visit", default="V1")
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    df = df[df["visit"].astype(str).str.upper() == args.visit.upper()].copy()

    rows, skipped = [], []
    for r in df.itertuples(index=False):
        sid = getattr(r, args.id_col)

        if not isinstance(r.wmh_gt_path, str) or not r.wmh_gt_path:
            skipped.append((sid, "missing WMH reference"))
            continue
        if not isinstance(r.wmh_pred_path, str) or not r.wmh_pred_path:
            skipped.append((sid, "missing WMH prediction"))
            continue

        gt = load_bool(r.wmh_gt_path)
        pr = load_bool(r.wmh_pred_path)

        bg = load_bool(r.left_BG_path) | load_bool(r.right_BG_path)
        cso = load_bool(r.left_CSO_path) | load_bool(r.right_CSO_path)
        pvs_total = bg | cso

        if not (gt.shape == pr.shape == bg.shape == cso.shape):
            skipped.append((sid, f"shape mismatch gt={gt.shape} pred={pr.shape} bg={bg.shape}"))
            continue

        whole = np.ones_like(gt, dtype=bool)

        c_whole = derive_metrics(confusion(gt, pr, whole))
        c_bg = derive_metrics(confusion(gt, pr, bg))
        c_cso = derive_metrics(confusion(gt, pr, cso))
        c_pvs = derive_metrics(confusion(gt, pr, pvs_total))

        fp_whole = c_whole["FP"]

        row = {"subject_id": sid, "visit": args.visit}
        row |= pack("whole", c_whole)
        row |= pack("BG", c_bg)
        row |= pack("CSO", c_cso)
        row |= pack("PVS", c_pvs)
        row["BG_FP_concentration"] = safe_rate(c_bg["FP"], fp_whole)
        row["CSO_FP_concentration"] = safe_rate(c_cso["FP"], fp_whole)
        row["PVS_FP_concentration"] = safe_rate(c_pvs["FP"], fp_whole)
        # ROI sizes in voxels, so the FP-per-mL normalisation downstream is auditable.
        row["BG_ROI_voxels"] = int(bg.sum())
        row["CSO_ROI_voxels"] = int(cso.sum())

        rows.append(row)

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print("Wrote:", args.out_csv)
    print("Rows:", len(out))
    if skipped:
        print(f"\nSkipped {len(skipped)} participant(s):")
        for sid, why in skipped:
            print(f"  {sid}: {why}")
        print("Skipped participants are one reason n differs between analyses. "
              "See statistics/README.md.")


if __name__ == "__main__":
    main()
