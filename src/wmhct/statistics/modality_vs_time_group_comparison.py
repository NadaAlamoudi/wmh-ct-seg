"""
Manuscript: Section 3.5 - scanning modality against time interval: group comparison.

Run: wmhct-modality-comparison --help

Note: Pass --sheet For_statistical_analyses for the published 218-row sample.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import levene, mannwhitneyu, shapiro, ttest_ind  # noqa: E402

VOLUME_COLUMNS = ["WMHstable_vol", "WMHshrink_vol", "WMHgrow_vol"]
COUNT_COLUMNS = ["WMHstable_count", "WMHshrink_count", "WMHgrow_count"]
# Variables Section 3.5 compares in addition to the six WMH change outcomes. Column
# names follow the curated analysis sheet For_statistical_analyses.
PAPER_EXTRA_VARIABLES = (
    "Time_Difference_Days",   # inter-scan interval
    "Age",
    "Sex",
    "indexSLvol_mm3",         # index stroke lesion
    "oldSLvol_mm3",           # old stroke lesion
    "newSLvolV1_mm3",         # incident infarct at V1
)

GROUPS = ("CT-MRI", "MRI-MRI")


def holm_correct(pvalues):
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


def rank_biserial_from_u(u, n1, n2):
    """Rank-biserial correlation, which needs no normal approximation and no tie fix."""
    return float(2 * u / (n1 * n2) - 1)


def r_from_z_no_tie_correction(u, n1, n2):
    """The original effect size: |z| / sqrt(N), normal approximation, ties ignored."""
    mean_u = n1 * n2 / 2
    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u - mean_u) / std_u
    return float(abs(z) / np.sqrt(n1 + n2))


def r_from_z_tie_corrected(u, a, b):
    """|z| / sqrt(N) with the standard tie correction applied to the variance of U."""
    n1, n2 = len(a), len(b)
    combined = np.concatenate([a, b])
    _, counts = np.unique(combined, return_counts=True)
    n = n1 + n2
    tie_term = np.sum(counts ** 3 - counts)
    var_u = (n1 * n2 / 12.0) * ((n + 1) - tie_term / (n * (n - 1))) if n > 1 else np.nan
    if not np.isfinite(var_u) or var_u <= 0:
        return np.nan
    z = (u - n1 * n2 / 2) / np.sqrt(var_u)
    return float(abs(z) / np.sqrt(n))


def cohens_d(a, b):
    n1, n2 = len(a), len(b)
    pooled = np.sqrt(((n1 - 1) * a.std(ddof=1) ** 2 + (n2 - 1) * b.std(ddof=1) ** 2)
                     / (n1 + n2 - 2))
    return float((a.mean() - b.mean()) / pooled) if pooled else np.nan


def main():
    parser = argparse.ArgumentParser(
        description="Section 3.5: CT-MRI versus MRI-MRI group comparison")
    parser.add_argument("--sheet", default=0,
                        help="worksheet name or index. The curated analysis table is "
                             "the sheet 'For_statistical_analyses' of "
                             "combined_data_for_analysis_and_paper1_MVH.xlsx (218 rows, "
                             "46 CT-MRI and 172 MRI-MRI), which is the sample the "
                             "manuscript reports. The default first sheet of that "
                             "workbook is All_measurements and is NOT that sample.")
    parser.add_argument("--combined_xlsx", required=True,
                        help="combined_data_for_analysis_corrected_allData.xlsx")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--group_col", default="Group")
    parser.add_argument("--id_col", default=None,
                        help="Participant identifier, if present, for the independence check")
    parser.add_argument("--test", choices=["auto", "mannwhitney", "ttest"], default="auto",
                        help="'auto' reproduces the original data-dependent selection")
    parser.add_argument("--correction", choices=["none", "holm"], default="none")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--extra_variables", nargs="*", default=list(PAPER_EXTRA_VARIABLES),
                        help="further columns to compare between the groups. Section 3.5 "
                             "reports the inter-scan interval, age, sex, index and old "
                             "stroke lesion volume and incident infarct volume alongside "
                             "the six WMH change outcomes, so those are the defaults. "
                             "Pass with no names to test the six outcomes only.")
    parser.add_argument("--no_wmh_difference", action="store_true",
                        help="do not derive WMH_volume_difference from WMH_vol_v1 minus "
                             "WMH_vol_v0. Section 3.5 reports that comparison.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)
    df = pd.read_excel(args.combined_xlsx, sheet_name=sheet)

    # Sex is compared with the same non-parametric test as everything else, per
    # Methods 2.8.3, so it has to be numeric first.
    for col in df.columns:
        if str(col).lower().startswith("sex") and df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip().str.upper().map(
                {"M": 0, "F": 1, "MALE": 0, "FEMALE": 1})

    # WMH volume difference between visits. Section 3.5 compares it between groups.
    if not args.no_wmh_difference and {"WMH_vol_v0", "WMH_vol_v1"}.issubset(df.columns):
        df["WMH_volume_difference"] = df["WMH_vol_v1"] - df["WMH_vol_v0"]

    requested = (VOLUME_COLUMNS + COUNT_COLUMNS
                 + (["WMH_volume_difference"] if "WMH_volume_difference" in df.columns
                    else [])
                 + list(args.extra_variables))
    seen, ordered = set(), []
    for c in requested:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    variables = [c for c in ordered if c in df.columns]
    missing = [c for c in ordered if c not in df.columns]
    if missing:
        print(f"WARNING: columns absent and skipped: {missing}")

    present_groups = set(df[args.group_col].dropna().unique())
    if not set(GROUPS).issubset(present_groups):
        raise ValueError(f"Expected groups {GROUPS} in '{args.group_col}'. Found: "
                         f"{sorted(present_groups)}")

    # Independence check.
    if args.id_col and args.id_col in df.columns:
        both = (df.groupby(args.id_col)[args.group_col].nunique() > 1).sum()
        if both:
            print(f"WARNING: {both} participant(s) appear in BOTH groups. The two samples "
                  f"are not independent and these p-values are optimistic. A paired or "
                  f"mixed-effects comparison would be required.")
        else:
            print("Independence check: no participant appears in both groups.")
    else:
        print("NOTE: no --id_col supplied, so independence between the groups was not "
              "checked. If a participant can contribute to both, the tests below are "
              "not valid as run.")

    descriptives = df.groupby(args.group_col)[variables].agg(
        ["count", "mean", "std", "median", "min", "max"])
    descriptives.to_csv(out_dir / "section_3_5_descriptives.csv")
    print("\nDescriptive statistics by group:")
    print(descriptives)

    rows = []
    for var in variables:
        a = df.loc[df[args.group_col] == GROUPS[0], var].dropna()
        b = df.loc[df[args.group_col] == GROUPS[1], var].dropna()
        if len(a) < 3 or len(b) < 3:
            print(f"Skipping {var}: too few observations ({len(a)}, {len(b)})")
            continue

        p_norm_a = shapiro(a).pvalue
        p_norm_b = shapiro(b).pvalue
        p_levene = levene(a, b).pvalue

        if args.test == "mannwhitney":
            use_parametric = False
        elif args.test == "ttest":
            use_parametric = True
        else:
            use_parametric = (p_norm_a > args.alpha) and (p_norm_b > args.alpha)

        if use_parametric:
            equal_var = p_levene > args.alpha
            stat, p_value = ttest_ind(a, b, equal_var=equal_var)
            test_name = ("Independent t-test (equal variances)" if equal_var
                         else "Welch t-test (unequal variances)")
            effect, effect_name = cohens_d(a, b), "Cohen's d"
            effect_alt = np.nan
        else:
            stat, p_value = mannwhitneyu(a, b, alternative="two-sided")
            test_name = "Mann-Whitney U"
            effect = r_from_z_no_tie_correction(stat, len(a), len(b))
            effect_name = "r = |z|/sqrt(N), ties ignored (as originally computed)"
            effect_alt = r_from_z_tie_corrected(stat, a.values, b.values)

        rows.append({
            "Variable": var,
            "n_CT_MRI": len(a), "n_MRI_MRI": len(b),
            "median_CT_MRI": float(a.median()), "median_MRI_MRI": float(b.median()),
            "mean_CT_MRI": float(a.mean()), "mean_MRI_MRI": float(b.mean()),
            "shapiro_p_CT_MRI": float(p_norm_a), "shapiro_p_MRI_MRI": float(p_norm_b),
            "levene_p": float(p_levene),
            "Test": test_name, "Statistic": float(stat), "p_raw": float(p_value),
            "EffectSize": effect, "EffectSizeType": effect_name,
            "EffectSize_tie_corrected": effect_alt,
            "rank_biserial": (rank_biserial_from_u(stat, len(a), len(b))
                              if test_name == "Mann-Whitney U" else np.nan),
        })

        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=var, hue=args.group_col, kde=True)
        plt.title(f"Distribution of {var} by group")
        plt.tight_layout()
        plt.savefig(out_dir / f"dist_{var}.png", dpi=200)
        plt.close()

    results = pd.DataFrame(rows)
    if results.empty:
        print("No variable had enough data to test.")
        return

    if args.correction == "holm":
        results["p_adjusted"] = holm_correct(results["p_raw"].values)
        decision = "p_adjusted"
    else:
        results["p_adjusted"] = np.nan
        decision = "p_raw"
    results["correction"] = args.correction
    results["Significant"] = results[decision] < args.alpha

    out_csv = out_dir / "section_3_5_group_comparison.csv"
    results.to_csv(out_csv, index=False)

    print("\nGroup comparison (CT-MRI vs MRI-MRI):")
    print(results[["Variable", "Test", "Statistic", "p_raw", "p_adjusted",
                   "EffectSize", "Significant"]].to_string(index=False))
    print(f"\nWrote {out_csv}")
    print(f"Significant at alpha={args.alpha} ({args.correction}): "
          f"{int(results['Significant'].sum())} of {len(results)}")
    if args.correction == "none":
        n_holm = int((holm_correct(results['p_raw'].values) < args.alpha).sum())
        print(f"For comparison, under Holm-Bonferroni it would be {n_holm} of "
              f"{len(results)}. Re-run with --correction holm to report that version.")


if __name__ == "__main__":
    main()
