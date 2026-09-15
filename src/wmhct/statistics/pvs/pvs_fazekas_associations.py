"""
Section 3.8 - PVS density against regional false positives, and Fazekas against WMH burden.

Manuscript: the PVS association analysis and the Fazekas correlation
Run: wmhct-pvs-associations --help
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import linregress, mannwhitneyu, spearmanr  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402

PVS_ID_COL_DEFAULT = "MSS3_Study_Number"
WMH_ID_COL_DEFAULT = "subject_id"

PVS_PAIRS = [
    ("BG_PVS_density_pct", "BG_FP_per_mL"),
    ("CSO_PVS_density_pct", "CSO_FP_per_mL"),
    ("BG_PVS_density_pct", "BG_FDR"),
    ("CSO_PVS_density_pct", "CSO_FDR"),
    ("BG_PVS_density_pct", "BG_FP_concentration"),
    ("CSO_PVS_density_pct", "CSO_FP_concentration"),
    ("Overall_PVS_density_pct", "PVS_FP_concentration"),
    ("BG_PVS_density_pct", "BG_FP"),
    ("CSO_PVS_density_pct", "CSO_FP"),
]

FAZEKAS_COLS = ["V1_PVLOverall", "V1_DWMLOverall"]
BURDEN_COLS = ["Pred_WMH_vox", "GT_WMH_vox"]


def norm_id(s):
    return s.astype(str).str.strip().str.upper()


def holm_correct(pvalues):
    """Holm-Bonferroni adjusted p-values, order preserved. NaNs pass through."""
    p = np.asarray(pvalues, dtype=float)
    finite = ~np.isnan(p)
    adjusted = np.full_like(p, np.nan)
    idx = np.where(finite)[0]
    n = len(idx)
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (n - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted


def spearman_report(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan, np.nan, int(ok.sum())
    rho, p = spearmanr(x[ok], y[ok])
    return float(rho), float(p), int(ok.sum())


def linreg_r2(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3:
        return np.nan, int(ok.sum())
    lr = LinearRegression().fit(x[ok].to_numpy().reshape(-1, 1), y[ok].to_numpy())
    return float(lr.score(x[ok].to_numpy().reshape(-1, 1), y[ok].to_numpy())), int(ok.sum())


def median_split_mwu(x, y):
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 6:
        return None
    x_ok, y_ok = x[ok], y[ok]
    med = float(np.nanmedian(x_ok))
    low, high = y_ok[x_ok <= med], y_ok[x_ok > med]
    if len(low) < 2 or len(high) < 2:
        return None
    u, p = mannwhitneyu(low, high, alternative="two-sided")
    return {"n": int(ok.sum()), "median_split_at": med, "n_low": len(low), "n_high": len(high),
            "median_low": float(np.nanmedian(low)), "median_high": float(np.nanmedian(high)),
            "U": float(u), "p_raw": float(p)}


def scatter_with_spearman(df, xcol, ycol, xlabel, ylabel, out_png, title=None):
    x = pd.to_numeric(df[xcol], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    ok = x.notna() & y.notna()

    rho, p = spearmanr(x[ok], y[ok]) if ok.sum() >= 3 else (np.nan, np.nan)
    n = int(ok.sum())

    plt.figure(figsize=(14, 10))
    plt.scatter(x[ok], y[ok], s=35)
    ax = plt.gca()

    if ok.sum() >= 2:
        lr = linregress(x[ok].values, y[ok].values)
        xs = np.linspace(x[ok].min(), x[ok].max(), 200)
        ax.plot(xs, lr.intercept + lr.slope * xs, linewidth=2)

    if title:
        ax.set_title(title, fontsize=26, pad=12)
    ax.set_xlabel(xlabel, fontsize=22, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=22, labelpad=10)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.text(0.02, 0.98, f"Spearman rho = {rho:.3f}\np = {p:.3g}\nn = {n}",
            transform=ax.transAxes, va="top", ha="left", fontsize=22,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.9))

    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()


def box_by_grade(df, grade_col, ycol, title, out_png):
    g = pd.to_numeric(df[grade_col], errors="coerce")
    y = pd.to_numeric(df[ycol], errors="coerce")
    ok = g.notna() & y.notna()
    d = pd.DataFrame({grade_col: g[ok], ycol: y[ok]})
    if d.empty:
        return

    grades = sorted(d[grade_col].unique())
    data = [d.loc[d[grade_col] == gr, ycol].to_numpy() for gr in grades]

    plt.figure(figsize=(12, 6))
    plt.boxplot(data, labels=[str(gr) for gr in grades])
    rng = np.random.default_rng(0)
    for i, gr in enumerate(grades, start=1):
        vals = d.loc[d[grade_col] == gr, ycol].to_numpy()
        jitter = (rng.random(len(vals)) - 0.5) * 0.15
        plt.scatter(np.full_like(vals, i, dtype=float) + jitter, vals)

    plt.xlabel(f"{grade_col} (reader-assigned grade)")
    plt.ylabel(ycol)
    plt.title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="PVS density and Fazekas grade against WMH burden")
    parser.add_argument("--pvs_csv", required=True, help="pvs_density_v1.csv")
    parser.add_argument("--wmh_metrics_csv", required=True, help="wmh_metrics_by_roi_V1.csv")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--pvs_id_col", default=PVS_ID_COL_DEFAULT)
    parser.add_argument("--wmh_id_col", default=WMH_ID_COL_DEFAULT)
    parser.add_argument("--visit", default="V1")
    parser.add_argument("--correction", choices=["none", "holm"], default="none",
                        help="Holm-Bonferroni within each family. 'none' reproduces the original")
    parser.add_argument("--median_split", action="store_true",
                        help="Also run the median-split Mann-Whitney comparison, which the "
                             "original computed but discarded")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pvs = pd.read_csv(args.pvs_csv)
    wmh = pd.read_csv(args.wmh_metrics_csv)

    for name, df, col in [("PVS", pvs, args.pvs_id_col), ("WMH", wmh, args.wmh_id_col)]:
        if col not in df.columns:
            raise ValueError(f"{name} ID column '{col}' not found. Available: {list(df.columns)}")

    pvs["ID"] = norm_id(pvs[args.pvs_id_col])
    wmh["ID"] = norm_id(wmh[args.wmh_id_col])

    for df in (pvs, wmh):
        if "visit" in df.columns:
            df.drop(df.index[df["visit"].astype(str).str.upper() != args.visit.upper()],
                    inplace=True)

    merged = pvs.merge(wmh, on="ID", how="inner", suffixes=("_pvs", "_wmh"))
    merged.to_csv(out_dir / "merged_pvs_wmh.csv", index=False)

    missing_in_wmh = sorted(set(pvs["ID"]) - set(wmh["ID"]))
    missing_in_pvs = sorted(set(wmh["ID"]) - set(pvs["ID"]))
    pd.DataFrame({"ID": missing_in_wmh}).to_csv(out_dir / "missing_in_wmh.csv", index=False)
    pd.DataFrame({"ID": missing_in_pvs}).to_csv(out_dir / "missing_in_pvs.csv", index=False)

    coverage = (f"PVS rows: {len(pvs)} | WMH rows: {len(wmh)} | merged: {len(merged)}\n"
                f"In PVS only (no WMH metrics): {len(missing_in_wmh)}\n"
                f"In WMH only (no PVS density): {len(missing_in_pvs)}")
    print(coverage)
    (out_dir / "merge_coverage.txt").write_text(coverage + "\n")

    for col in ["whole_TP", "whole_FP", "whole_FN"]:
        if col not in merged.columns:
            raise ValueError(f"Expected column {col} not found. Available: {list(merged.columns)}")

    merged["Pred_WMH_vox"] = merged["whole_TP"] + merged["whole_FP"]
    merged["GT_WMH_vox"] = merged["whole_TP"] + merged["whole_FN"]
    merged["BG_FP_per_mL"] = merged["BG_FP"] / (merged["V1_BG_ROIvol_mm3"] / 1000.0)
    merged["CSO_FP_per_mL"] = merged["CSO_FP"] / (merged["V1_CSO_ROIvol_mm3"] / 1000.0)

    # ---------------- Family A: PVS density vs false-positive burden ----------------
    rows_a = []
    for xcol, ycol in PVS_PAIRS:
        if xcol not in merged.columns or ycol not in merged.columns:
            continue
        rho, p, n = spearman_report(merged[xcol], merged[ycol])
        rows_a.append({"family": "A_PVS_vs_FP", "x": xcol, "y": ycol,
                       "spearman_rho": rho, "p_raw": p, "n": n})

    # ---------------- Family B: Fazekas grade vs WMH burden -------------------------
    rows_b = []
    for faz_col in FAZEKAS_COLS:
        if faz_col not in merged.columns:
            print(f"NOTE: {faz_col} not present, skipping that Fazekas comparison.")
            continue
        for ycol in BURDEN_COLS:
            rho, p, n = spearman_report(merged[faz_col], merged[ycol])
            r2, n2 = linreg_r2(merged[faz_col], merged[ycol])
            rows_b.append({
                "family": "B_Fazekas_vs_burden", "x": faz_col, "y": ycol,
                "spearman_rho": rho, "p_raw": p, "n": n,
                "linreg_R2_ordinal_predictor_use_with_care": r2, "n_linreg": n2,
            })
            box_by_grade(merged, faz_col, ycol,
                         f"{ycol} by {faz_col} (Spearman rho={rho:.3f}, p={p:.3g}, n={n})",
                         out_dir / f"box_{ycol}_by_{faz_col}.png")

    # Correction is applied within each family, not across both.
    frames = []
    for rows in (rows_a, rows_b):
        if not rows:
            continue
        t = pd.DataFrame(rows)
        t["p_adjusted"] = holm_correct(t["p_raw"].values) if args.correction == "holm" else np.nan
        t["correction"] = args.correction
        frames.append(t)

    if args.median_split:
        ms_rows = []
        for xcol, ycol in [("BG_PVS_density_pct", "BG_FP_per_mL"),
                           ("CSO_PVS_density_pct", "CSO_FP_per_mL")]:
            if xcol in merged.columns and ycol in merged.columns:
                res = median_split_mwu(merged[xcol], merged[ycol])
                if res:
                    ms_rows.append({"family": "A_median_split", "x": xcol, "y": ycol, **res})
        if ms_rows:
            frames.append(pd.DataFrame(ms_rows))

    stats_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    stats_out = out_dir / "association_stats_summary.csv"
    stats_df.to_csv(stats_out, index=False)

    # ---------------- Key figures --------------------------------------------------
    for xcol, ycol, xlabel, ylabel, fname, title in [
        ("BG_PVS_density_pct", "BG_FP_per_mL",
         "PVS density in BG ROI (%)", "WMH false positives per mL of BG ROI",
         "Fig_BG_PVSdensity_vs_BG_FPperML.png",
         "PVS density vs WMH false-positive burden, BG ROI"),
        ("CSO_PVS_density_pct", "CSO_FP_per_mL",
         "PVS density in CSO ROI (%)", "WMH false positives per mL of CSO ROI",
         "Fig_CSO_PVSdensity_vs_CSO_FPperML.png",
         "PVS density vs WMH false-positive burden, CSO ROI"),
    ]:
        if xcol in merged.columns and ycol in merged.columns:
            scatter_with_spearman(merged, xcol, ycol, xlabel, ylabel, out_dir / fname, title)

    print("\nWrote:", stats_out)
    if not stats_df.empty:
        print(stats_df.to_string(index=False))
        ns = sorted(stats_df["n"].dropna().unique())
        if len(ns) > 1:
            print(f"\nNOTE: n varies across analyses: {ns}. This is pairwise NaN dropping "
                  f"plus merge coverage, not an error. Quote the per-analysis n from the "
                  f"'n' column rather than a single cohort size.")


if __name__ == "__main__":
    main()
