"""
Section 3.7 - effect of stroke lesions on WMH segmentation accuracy.

Manuscript: Section 3.7, "Impact of stroke lesions on WMH segmentation"
Run: wmhct-stroke-dice --help

"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from scipy import stats  # noqa: E402


def clean_id(x):
    """Reduce either identifier convention to one canonical key."""
    digits = re.findall(r"\d+", str(x))
    return "".join(digits) if digits else str(x).strip()


def join_model7_dice(data, metrics_xlsx, id_col, dice_col):
    """Replace the Dice column with the verified Model 7 values, matched on scan ID."""
    m7 = pd.read_excel(metrics_xlsx, sheet_name="All_Folds")
    m7["_id"] = m7["Study_ID"].apply(clean_id)
    data["_id"] = data[id_col].apply(clean_id)

    before = len(data)
    merged = data.merge(m7[["_id", "Dice"]], on="_id", how="inner")
    print(f"Model 7 join: {len(merged)} of {before} rows matched.")
    unmatched = sorted(set(data["_id"]) - set(m7["_id"]))
    if unmatched:
        print(f"  unmatched scan IDs dropped: {unmatched[:10]}")

    if merged.empty:
        raise SystemExit(
            "The Model 7 join matched no rows, so there is nothing to analyse.\n"
            f"  stroke sheet IDs look like: {sorted(data['_id'])[:3]}\n"
            f"  metrics sheet IDs look like: {sorted(m7['_id'])[:3]}\n"
            "Both should reduce to a digit string such as '028' or '0281'. "
            "Check that --model7_metrics points at the per-case workbook whose "
            "All_Folds sheet has a Study_ID column.")

    if dice_col in merged.columns:
        diff = (merged[dice_col] - merged["Dice"]).abs()
        n_diff = int((diff > 1e-6).sum())
        print(f"  rows where the file's Dice differs from Model 7: {n_diff} of {len(merged)} "
              f"(max difference {diff.max():.4f})")
    merged[dice_col] = merged["Dice"]
    return merged.drop(columns=["Dice", "_id"])


def fit_and_report(formula, data, label, out_dir, lines):
    try:
        model = smf.ols(formula, data=data).fit()
    except Exception as e:
        lines.append(f"{label}: model failed to fit ({e})")
        print(lines[-1])
        return None
    lines.append(f"\n=== {label} ===\n{formula}\n{model.summary()}\n")
    print(lines[-1])

    rows = []
    for name in model.params.index:
        rows.append({"model": label, "term": name,
                     "coef": float(model.params[name]),
                     "std_err": float(model.bse[name]),
                     "p": float(model.pvalues[name]),
                     "ci_low": float(model.conf_int().loc[name, 0]),
                     "ci_high": float(model.conf_int().loc[name, 1]),
                     "R_squared": float(model.rsquared), "n": int(model.nobs)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Section 3.7: effect of stroke lesions on WMH segmentation accuracy")
    parser.add_argument("--stroke_xlsx", required=True,
                        help="updated_results_with_dice_volumes_stroke_info.xlsx")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dice_col", default="Dice_Coefficient_CT",
                        help="Section 3.7 quantifies agreement in CT space, where the "
                             "prediction is native, so the CT column is the default. "
                             "The superseded workbook has Dice_Coefficient_FLAIR instead.")
    parser.add_argument("--id_col", default="Study_ID")
    parser.add_argument("--wmh_col", default="WMHvol")
    parser.add_argument("--old_vol_col", default="oldSLvol_r")
    parser.add_argument("--index_vol_col", default="indexSLvo")
    parser.add_argument("--old_type_col", default="oldStrokeType")
    parser.add_argument("--index_type_col", default="indexStrokeType")
    parser.add_argument("--model7_metrics", default=None,
                        help="Dataset051_..._metrics.xlsx, to replace the Dice column "
                             "with the verified Model 7 values")
    parser.add_argument("--acknowledge_dice_source", action="store_true",
                        help="Run on the file's own Dice column, accepting that it was "
                             "diagnosed as Model 6")
    parser.add_argument("--standardise", action="store_true")
    parser.add_argument("--log_volumes", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_excel(args.stroke_xlsx)

    if args.model7_metrics:
        data = join_model7_dice(data, args.model7_metrics, args.id_col, args.dice_col)
        dice_provenance = "Model 7, joined from the verified metrics workbook"
    elif args.acknowledge_dice_source:
        dice_provenance = ("the input workbook, diagnosed during the revision as Model 6, "
                           "not Model 7")
        print("WARNING: running on the workbook's own Dice column. It was diagnosed as "
              "Model 6. See the module docstring.")
    else:
        raise SystemExit(
            "Refusing to run on an unverified Dice column.\n"
            "The Dice values in updated_results_with_dice_volumes_stroke_info.xlsx were\n"
            "diagnosed as Model 6, while the manuscript reports Model 7.\n\n"
            "Either:\n"
            "  --model7_metrics /path/to/Dataset051_..._metrics.xlsx   (recommended)\n"
            "  --acknowledge_dice_source                               (run as-is)\n")

    numeric = [args.dice_col, args.wmh_col, args.old_vol_col, args.index_vol_col]
    for c in numeric:
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not found. Available: {list(data.columns)}")
        data[c] = pd.to_numeric(data[c], errors="coerce")

    desc = data[numeric].describe().T
    desc.to_csv(out_dir / "section_3_7_descriptives.csv")
    print("\nDescriptive statistics:")
    print(desc)

    # ---- correlations ----------------------------------------------------------
    corr_rows = []
    for a, b in [(args.dice_col, args.old_vol_col), (args.dice_col, args.index_vol_col),
                 (args.dice_col, args.wmh_col), (args.old_vol_col, args.wmh_col),
                 (args.index_vol_col, args.wmh_col)]:
        sub = data[[a, b]].dropna()
        if len(sub) < 3:
            continue
        pr, pp = stats.pearsonr(sub[a], sub[b])
        sr, sp = stats.spearmanr(sub[a], sub[b])
        corr_rows.append({"x": a, "y": b, "n": len(sub),
                          "pearson_r": pr, "pearson_p": pp,
                          "spearman_rho": sr, "spearman_p": sp})
    corr = pd.DataFrame(corr_rows)
    corr.to_csv(out_dir / "section_3_7_correlations.csv", index=False)
    print("\nCorrelations (Spearman is the one to quote; volumes are skewed):")
    print(corr.to_string(index=False))

    plt.figure(figsize=(9, 7))
    sns.heatmap(data[numeric].corr(method="spearman"), annot=True, cmap="coolwarm",
                vmin=-1, vmax=1)
    plt.title("Spearman correlation of key variables")
    plt.tight_layout()
    plt.savefig(out_dir / "section_3_7_correlation_matrix.png", dpi=200)
    plt.close()

    # ---- regression models -----------------------------------------------------
    model_data = data.copy()
    if args.standardise:
        for c in numeric:
            sd = model_data[c].std()
            model_data[c] = (model_data[c] - model_data[c].mean()) / sd if sd else np.nan
        scale_note = "standardised (z-scored) predictors and outcome"
    else:
        scale_note = "raw scales; volume coefficients are per mm3"

    lines, frames = [], []
    lines.append(f"Dice provenance: {dice_provenance}")
    lines.append(f"Coefficient scale: {scale_note}")

    frames.append(fit_and_report(
        f"Q('{args.dice_col}') ~ Q('{args.old_vol_col}') + C(Q('{args.old_type_col}')) "
        f"+ Q('{args.wmh_col}')", model_data, "Old stroke lesion", out_dir, lines))
    frames.append(fit_and_report(
        f"Q('{args.dice_col}') ~ Q('{args.index_vol_col}') + C(Q('{args.index_type_col}')) "
        f"+ Q('{args.wmh_col}')", model_data, "Index stroke lesion", out_dir, lines))

    if args.log_volumes:
        model_data["log_old"] = np.log1p(model_data[args.old_vol_col])
        model_data["log_index"] = np.log1p(model_data[args.index_vol_col])
        frames.append(fit_and_report(
            f"Q('{args.dice_col}') ~ log_old + C(Q('{args.old_type_col}')) + Q('{args.wmh_col}')",
            model_data, "Old stroke lesion, log volume", out_dir, lines))
        frames.append(fit_and_report(
            f"Q('{args.dice_col}') ~ log_index + C(Q('{args.index_type_col}')) + Q('{args.wmh_col}')",
            model_data, "Index stroke lesion, log volume", out_dir, lines))

    coefs = pd.concat([f for f in frames if f is not None], ignore_index=True)
    coefs.to_csv(out_dir / "section_3_7_regression_coefficients.csv", index=False)
    (out_dir / "section_3_7_model_summaries.txt").write_text("\n".join(lines))

    # ---- figures ---------------------------------------------------------------
    for vol_col, type_col, tag in [(args.old_vol_col, args.old_type_col, "old"),
                                   (args.index_vol_col, args.index_type_col, "index")]:
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        sns.scatterplot(data=data, x=vol_col, y=args.dice_col,
                        hue=type_col if type_col in data.columns else None, ax=ax[0])
        ax[0].set_title(f"Dice vs {tag} stroke lesion volume by type")
        ax[0].set_xlabel(f"{tag} stroke lesion volume (mm3)")
        ax[0].set_ylabel("Dice")
        if type_col in data.columns:
            sns.boxplot(data=data, x=type_col, y=args.dice_col, ax=ax[1])
            ax[1].set_title(f"Dice by {tag} stroke type")
        plt.tight_layout()
        plt.savefig(out_dir / f"section_3_7_dice_vs_{tag}_stroke.png", dpi=200)
        plt.close(fig)

    print(f"\nWrote outputs to {out_dir}")
    print(f"Dice provenance recorded as: {dice_provenance}")


if __name__ == "__main__":
    main()
