"""
Figures 8 and 9 - paired per-case differences between Model 5 and Model 7.

Manuscript: Dice and MAE difference panels by WMH volume class
Run: wmhct-model5-vs-model7 --help
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.stats as stats  # noqa: E402
import seaborn as sns  # noqa: E402

CLASS_ORDER = ['Mild', 'Moderate', 'Severe']
LOW_MAX_ML = 10.0
MEDIUM_MAX_ML = 25.0


def clean_study_id(x):
    x_str = str(x).strip()
    if x_str.startswith("MSS3_"):
        x_str = x_str.replace("MSS3_", "")
    try:
        return str(int(x_str))
    except ValueError:
        return x_str


def classify_burden(volume_ml):
    if volume_ml <= LOW_MAX_ML:
        return 'Mild'
    if volume_ml <= MEDIUM_MAX_ML:
        return 'Moderate'
    return 'Severe'


def paired_test(a, b, test):
    if len(a) < 2:
        return np.nan, np.nan
    if test == "wilcoxon":
        if np.allclose(np.asarray(a) - np.asarray(b), 0):
            return np.nan, 1.0
        return stats.wilcoxon(a, b)
    return stats.ttest_rel(a, b)


def difference_plot(df, column, ylabel, title, out_path):
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='Burden_Class', y=column, data=df, palette="Set2", order=CLASS_ORDER)
    plt.title(title)
    plt.xlabel("WMH volume class")
    plt.ylabel(ylabel)
    plt.axhline(0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-case Model 5 versus Model 7 differences by burden class")
    parser.add_argument("--model5_xlsx", required=True, help="Dataset034 metrics")
    parser.add_argument("--model7_xlsx", required=True, help="Dataset051 metrics")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--test", choices=["ttest", "wilcoxon"], default="wilcoxon")
    parser.add_argument("--dice_threshold", type=float, default=0.05,
                        help="Flag cases whose absolute Dice difference exceeds this")
    parser.add_argument("--mae_threshold", type=float, default=5.0,
                        help="Flag cases whose absolute MAE difference exceeds this, mL")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    m5 = pd.read_excel(args.model5_xlsx, sheet_name="All_Folds")
    m7 = pd.read_excel(args.model7_xlsx, sheet_name="All_Folds")

    m5["ID"] = m5["Study_ID"].apply(clean_study_id)
    m7["ID"] = m7["Study_ID"].apply(clean_study_id)

    only_5 = sorted(set(m5["ID"]) - set(m7["ID"]))
    only_7 = sorted(set(m7["ID"]) - set(m5["ID"]))
    if only_5 or only_7:
        print(f"WARNING: unmatched scans. Only in Model 5: {only_5}. "
              f"Only in Model 7: {only_7}. These are dropped from the comparison.")

    merged = m5.merge(m7, on="ID", suffixes=("_m5", "_m7"))
    print(f"Matched {len(merged)} scans")

    comparison = pd.DataFrame({
        "ID": merged["ID"],
        "GT_Volume_ml": merged["GT_Volume_ml_m5"],
        "Dice_m5": merged["Dice_m5"],
        "Dice_m7": merged["Dice_m7"],
        "MAE_m5": merged["MAE(ml)_m5"],
        "MAE_m7": merged["MAE(ml)_m7"],
    })
    comparison["Dice_Diff"] = comparison["Dice_m7"] - comparison["Dice_m5"]
    comparison["MAE_Diff"] = comparison["MAE_m7"] - comparison["MAE_m5"]
    comparison["Burden_Class"] = comparison["GT_Volume_ml"].apply(classify_burden)

    # Paired tests: overall and within each burden class.
    rows = []
    for label, subset in [("All", comparison)] + [
            (cls, comparison[comparison["Burden_Class"] == cls]) for cls in CLASS_ORDER]:
        if subset.empty:
            continue
        for metric, a_col, b_col in [("Dice", "Dice_m7", "Dice_m5"),
                                     ("MAE", "MAE_m7", "MAE_m5")]:
            pair = subset[[a_col, b_col]].dropna()
            _, p = paired_test(pair[a_col], pair[b_col], args.test)
            diff = pair[a_col] - pair[b_col]
            sd = diff.std(ddof=1)
            rows.append({
                "Metric": metric,
                "Burden_Class": label,
                "n": len(pair),
                "Median_Difference": diff.median(),
                "Mean_Difference": diff.mean(),
                "Cohens_d": diff.mean() / sd if sd else np.nan,
                "p": p,
                "test": args.test,
            })

    results = pd.DataFrame(rows)
    print("\nPaired comparison, Model 7 minus Model 5:")
    print(results.to_string(index=False))
    results.to_csv(os.path.join(args.out_dir, "model5_vs_model7_tests.csv"), index=False)
    print("\nNo multiple-comparison correction is applied across these eight tests. "
          "Interpret accordingly.")

    difference_plot(
        comparison, "Dice_Diff", "Dice difference (Model 7 minus Model 5)",
        "Difference in Dice between Model 5 and Model 7 by WMH volume class\n"
        "(Mild <= 10 mL, Moderate <= 25 mL, Severe > 25 mL)",
        os.path.join(args.out_dir, "Dice_Difference_by_WMH_Volume_Class.png"))

    difference_plot(
        comparison, "MAE_Diff", "MAE difference in mL (Model 7 minus Model 5)",
        "Difference in MAE between Model 5 and Model 7 by WMH volume class\n"
        "(Mild <= 10 mL, Moderate <= 25 mL, Severe > 25 mL)",
        os.path.join(args.out_dir, "MAE_Difference_by_WMH_Volume_Class.png"))

    flagged = comparison[(comparison["Dice_Diff"].abs() > args.dice_threshold)
                         | (comparison["MAE_Diff"].abs() > args.mae_threshold)]
    flagged_path = os.path.join(args.out_dir, "cases_flagged_for_inspection.csv")
    flagged.to_csv(flagged_path, index=False)
    print(f"\n{len(flagged)} case(s) exceed the inspection thresholds "
          f"(|Dice diff| > {args.dice_threshold} or |MAE diff| > {args.mae_threshold} mL) "
          f"-> {flagged_path}")
    print("These thresholds are for visual triage only. They are not a significance test.")

    comparison.to_csv(os.path.join(args.out_dir, "model5_vs_model7_per_case.csv"),
                      index=False)


if __name__ == "__main__":
    main()
