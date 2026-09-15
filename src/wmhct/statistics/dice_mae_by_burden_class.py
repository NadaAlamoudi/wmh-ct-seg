"""
Figures 6 and 7 - Dice and MAE by WMH burden class for Models 5, 6 and 7.

Manuscript: Figures 6 and 7, Dice and MAE by WMH volume class
Run: wmhct-burden-class-figures --help
"""

import argparse
import os
from itertools import combinations

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
    """Standardise Study_ID: strip an 'MSS3_' prefix and drop leading zeros."""
    x_str = str(x).strip()
    if x_str.startswith("MSS3_"):
        x_str = x_str.replace("MSS3_", "")
    try:
        return str(int(x_str))
    except ValueError:
        return x_str


def classify_burden(volume_ml):
    """Volume-derived burden class. Not a reader-assigned Fazekas grade."""
    if volume_ml <= LOW_MAX_ML:
        return 'Mild'
    if volume_ml <= MEDIUM_MAX_ML:
        return 'Moderate'
    return 'Severe'


def load_model_metrics(model_files):
    dfs = {}
    for model_name, file_path in model_files.items():
        df = pd.read_excel(file_path, sheet_name="All_Folds")
        if "Study_ID" not in df.columns:
            raise KeyError(f"Column 'Study_ID' not found in {file_path}")
        df["Study_ID_clean"] = df["Study_ID"].apply(clean_study_id)
        df = df[["Study_ID_clean", "GT_Volume_ml", "MAE(ml)", "Dice"]].copy()
        dfs[model_name] = df.rename(columns={
            "GT_Volume_ml": f"GT_Volume_ml_{model_name}",
            "MAE(ml)": f"MAE_{model_name}",
            "Dice": f"Dice_{model_name}",
        })
    return dfs


def paired_test(a, b, test):
    if test == "wilcoxon":
        if np.allclose(np.asarray(a) - np.asarray(b), 0):
            return np.nan, 1.0
        return stats.wilcoxon(a, b)
    return stats.ttest_rel(a, b)


def holm_correct(pvalues):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = (n - rank) * p[idx]
        running = max(running, value)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def build_pairwise_table(merged_df, metric, models, test):
    """All pairwise comparisons for one metric, in every burden class."""
    rows = []
    for cls in CLASS_ORDER:
        subset = merged_df[merged_df["Burden_Class"] == cls]
        if subset.empty:
            continue
        for m1, m2 in combinations(models, 2):
            pair = subset[[f"{metric}_{m1}", f"{metric}_{m2}"]].dropna()
            if len(pair) < 2:
                continue
            a = pair[f"{metric}_{m1}"]
            b = pair[f"{metric}_{m2}"]
            _, p_val = paired_test(a, b, test)
            diff = a - b
            sd_diff = diff.std(ddof=1)
            rows.append({
                "Metric": metric,
                "Burden_Class": cls,
                "Model_A": m1,
                "Model_B": m2,
                "n": len(pair),
                "Mean_A": a.mean(),
                "Mean_B": b.mean(),
                "Mean_Difference": diff.mean(),
                "Cohens_d": diff.mean() / sd_diff if sd_diff != 0 else np.nan,
                "p_raw": p_val,
            })
    return pd.DataFrame(rows)


def make_figure(long_df, metric, ylabel, title, pairs, out_path, models,
                test, force_zero_ymin, ylim_top=None):
    sns.set_style("white")
    sns.set_context("talk", font_scale=1.2)

    plt.figure(figsize=(12, 8))
    ax = sns.boxplot(x='Burden_Class', y=metric, hue='Trainer', data=long_df,
                     palette="Set2", order=CLASS_ORDER)
    plt.title(title, fontsize=20)
    plt.xlabel("WMH volume class", fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    if ylim_top is not None:
        ax.set_ylim(0, ylim_top)
    plt.legend(title="Model", fontsize=16, title_fontsize=18)

    if pairs:
        try:
            from statannotations.Annotator import Annotator
        except ImportError:
            print("statannotations is not installed; writing the figure without "
                  "significance annotations. pip install statannotations")
        else:
            annotator = Annotator(ax, pairs=pairs, data=long_df, x='Burden_Class',
                                  y=metric, hue='Trainer', order=CLASS_ORDER,
                                  hue_order=models)
            annotator.configure(
                test='Wilcoxon' if test == 'wilcoxon' else 't-test_paired',
                text_format='star', loc='inside', verbose=2, fontsize=18)
            annotator.apply_and_annotate()

    if force_zero_ymin:
        _, ymax = ax.get_ylim()
        ax.set_ylim(0, ymax)

    sns.despine(offset=10, trim=True)
    plt.tight_layout(pad=2)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Dice and MAE by WMH burden class for Models 5, 6 and 7")
    parser.add_argument("--model5_xlsx", required=True,
                        help="Dataset034 (MSS3 only, predefined window, cropped, affine)")
    parser.add_argument("--model6_xlsx", required=True,
                        help="Dataset050 (fine-tuned on CIM pseudo-labels)")
    parser.add_argument("--model7_xlsx", required=True,
                        help="Dataset051 (additionally fine-tuned on IST-3)")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--test", choices=["ttest", "wilcoxon"], default="ttest",
                        help="ttest reproduces the original; wilcoxon is the safer choice")
    parser.add_argument("--correction", choices=["none", "holm"], default="none",
                        help="Multiple-comparison correction across the nine tests")
    parser.add_argument("--annotate", choices=["significant", "all"], default="significant",
                        help="'significant' reproduces the original")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    models = ['Model5', 'Model6', 'Model7']

    dfs = load_model_metrics({
        'Model5': args.model5_xlsx,
        'Model6': args.model6_xlsx,
        'Model7': args.model7_xlsx,
    })

    merged_df = dfs['Model5'].merge(dfs['Model6'], on="Study_ID_clean", how="left")
    merged_df = merged_df.merge(dfs['Model7'], on="Study_ID_clean", how="left")
    print(f"Total scans in merged dataset: {len(merged_df)}")

    merged_df["Burden_Class"] = merged_df["GT_Volume_ml_Model5"].apply(classify_burden)
    print(merged_df["Burden_Class"].value_counts().reindex(CLASS_ORDER))

    long_frames = {}
    for metric in ("Dice", "MAE"):
        long_df = pd.melt(
            merged_df,
            id_vars=["Study_ID_clean", "Burden_Class"],
            value_vars=[f"{metric}_{m}" for m in models],
            var_name="Trainer", value_name=metric,
        )
        long_df["Trainer"] = long_df["Trainer"].str.replace(f"{metric}_", "", regex=False)
        long_frames[metric] = long_df

    tables = []
    annotate_pairs = {}

    for metric in ("Dice", "MAE"):
        table = build_pairwise_table(merged_df, metric, models, args.test)
        if table.empty:
            annotate_pairs[metric] = []
            continue

        if args.correction == "holm":
            table["p_adjusted"] = holm_correct(table["p_raw"].values)
            decision_col = "p_adjusted"
        else:
            table["p_adjusted"] = np.nan
            decision_col = "p_raw"

        table["significant"] = table[decision_col] < args.alpha
        tables.append(table)

        if args.annotate == "all":
            selected = table
        else:
            selected = table[table["significant"]]

        annotate_pairs[metric] = [
            ((r.Burden_Class, r.Model_A), (r.Burden_Class, r.Model_B))
            for r in selected.itertuples()
        ]

        print(f"\n{metric}: {int(table['significant'].sum())} of {len(table)} "
              f"comparisons significant at alpha={args.alpha} "
              f"({args.correction} correction, {args.test})")

    if tables:
        all_tables = pd.concat(tables, ignore_index=True)
        table_path = os.path.join(args.out_dir, "pairwise_comparisons.csv")
        all_tables.to_csv(table_path, index=False)
        print(f"\nAll {len(all_tables)} pairwise comparisons written to {table_path}")

    make_figure(
        long_frames["Dice"], "Dice", "Dice coefficient",
        "Dice coefficient by WMH volume class\n"
        "(Mild <= 10 mL, Moderate <= 25 mL, Severe > 25 mL)",
        annotate_pairs.get("Dice", []),
        os.path.join(args.out_dir, "Dice_Coefficients_by_WMH_Volume_Class.png"),
        models, args.test, force_zero_ymin=True, ylim_top=1.0,
    )

    make_figure(
        long_frames["MAE"], "MAE",
        "Mean absolute error in WMH volume (mL)",
        "MAE in WMH volume by WMH volume class\n"
        "(Mild <= 10 mL, Moderate <= 25 mL, Severe > 25 mL)",
        annotate_pairs.get("MAE", []),
        os.path.join(args.out_dir, "MAE_WMH_Volume_Class_with_Stats.png"),
        models, args.test, force_zero_ymin=True,
        ylim_top=long_frames["MAE"]["MAE"].max() * 1.2,
    )


if __name__ == "__main__":
    main()
