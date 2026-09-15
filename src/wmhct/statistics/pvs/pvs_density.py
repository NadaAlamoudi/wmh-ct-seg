"""
Supporting analysis - PVS density per ROI, upstream of Section 3.8.

Manuscript: PVS ROI analysis; carries the reader-assigned Fazekas grades
Run: wmhct-pvs-density --help
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ESSENTIAL_COLS = [
    "V1_BG_PVS_vol_mm3", "V1_BG_PVS_count", "V1_BG_ROIvol_mm3",
    "V1_CSO_PVS_vol_mm3", "V1_CSO_PVS_count", "V1_CSO_ROIvol_mm3",
]
FAZEKAS_COLS = ["V1_PVLOverall", "V1_DWMLOverall"]

DENSITY_COLS = [
    "BG_PVS_density_pct", "BG_PVS_count_per_mL",
    "CSO_PVS_density_pct", "CSO_PVS_count_per_mL",
    "Overall_PVS_density_pct", "Overall_PVS_count_per_mL",
]


def safe_div(a, b):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    out = np.full_like(a, np.nan, dtype="float64")
    m = (b != 0) & (~np.isnan(b))
    out[m] = a[m] / b[m]
    return out


def summary_table(df, cols):
    def iqr(x):
        return np.nanpercentile(x, 75) - np.nanpercentile(x, 25)

    rows = []
    for c in cols:
        x = df[c].astype(float)
        rows.append({
            "variable": c,
            "n": int(np.sum(~np.isnan(x))),
            "mean": float(np.nanmean(x)),
            "sd": float(np.nanstd(x, ddof=1)),
            "median": float(np.nanmedian(x)),
            "iqr": float(iqr(x)),
            "min": float(np.nanmin(x)),
            "max": float(np.nanmax(x)),
        })
    return pd.DataFrame(rows)


def compute_densities(df):
    df = df.copy()
    for roi in ("BG", "CSO"):
        vol = df[f"V1_{roi}_PVS_vol_mm3"]
        cnt = df[f"V1_{roi}_PVS_count"]
        roivol = df[f"V1_{roi}_ROIvol_mm3"]
        df[f"{roi}_PVS_density_pct"] = safe_div(vol, roivol) * 100.0
        df[f"{roi}_PVS_count_per_mm3"] = safe_div(cnt, roivol)
        df[f"{roi}_PVS_count_per_100mm3"] = df[f"{roi}_PVS_count_per_mm3"] * 100.0
        df[f"{roi}_PVS_count_per_mL"] = df[f"{roi}_PVS_count_per_mm3"] * 1000.0

    total_roi = df["V1_BG_ROIvol_mm3"] + df["V1_CSO_ROIvol_mm3"]
    total_vol = df["V1_BG_PVS_vol_mm3"] + df["V1_CSO_PVS_vol_mm3"]
    total_cnt = df["V1_BG_PVS_count"] + df["V1_CSO_PVS_count"]

    df["Overall_PVS_density_pct"] = safe_div(total_vol, total_roi) * 100.0
    df["Overall_PVS_count_per_mm3"] = safe_div(total_cnt, total_roi)
    df["Overall_PVS_count_per_100mm3"] = df["Overall_PVS_count_per_mm3"] * 100.0
    df["Overall_PVS_count_per_mL"] = df["Overall_PVS_count_per_mm3"] * 1000.0
    return df


def main():
    parser = argparse.ArgumentParser(description="PVS density per ROI from the curated spreadsheet")
    parser.add_argument("--roi_manifest", required=True,
                        help="roi_manifest_V1.csv, defines which participants are included")
    parser.add_argument("--spreadsheet_xlsx", required=True,
                        help="Curated spreadsheet with PVS volumes, counts and Fazekas grades")
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--out_summary_csv", required=True)
    parser.add_argument("--roi_id_col", default="subject_id")
    parser.add_argument("--spreadsheet_id_col", default="MSS3_Study_Number")
    parser.add_argument("--sheet", default=0)
    args = parser.parse_args()

    roi = pd.read_csv(args.roi_manifest)
    v1_ids = set(roi[args.roi_id_col].astype(str).str.strip())

    df = pd.read_excel(args.spreadsheet_xlsx, sheet_name=args.sheet)

    missing = [c for c in [args.spreadsheet_id_col] + ESSENTIAL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Spreadsheet is missing required columns: {missing}\n"
                         f"Available: {list(df.columns)}")

    present_fazekas = [c for c in FAZEKAS_COLS if c in df.columns]
    if present_fazekas:
        print(f"Reader-assigned Fazekas columns found: {present_fazekas}")
    else:
        print("WARNING: neither V1_PVLOverall nor V1_DWMLOverall is present. The Fazekas "
              "correlation downstream will be skipped, and any reported rho against "
              "Fazekas grades cannot have come from this file.")

    df[args.spreadsheet_id_col] = df[args.spreadsheet_id_col].astype(str).str.strip()
    df_v1 = df[df[args.spreadsheet_id_col].isin(v1_ids)].copy()

    print(f"V1 IDs in ROI manifest:      {len(v1_ids)}")
    print(f"Rows matched in spreadsheet: {len(df_v1)}")
    unmatched = sorted(v1_ids - set(df_v1[args.spreadsheet_id_col]))
    if unmatched:
        print(f"WARNING: {len(unmatched)} manifest ID(s) absent from the spreadsheet: "
              f"{unmatched[:10]}")

    df_v1 = compute_densities(df_v1)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_v1.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

    summary_table(df_v1, DENSITY_COLS).to_csv(args.out_summary_csv, index=False)
    print("Wrote:", args.out_summary_csv)

    bad_roi = df_v1[(df_v1["V1_BG_ROIvol_mm3"] <= 0) | (df_v1["V1_CSO_ROIvol_mm3"] <= 0)]
    print("Non-positive ROI volumes (should be 0):", len(bad_roi))


if __name__ == "__main__":
    main()
