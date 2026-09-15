"""
Figure 5- fold composition by WMH burden category.

Run: wmhct-fold-composition --help
"""

import argparse
import os

import pandas as pd
from scipy.stats import chi2_contingency, hypergeom


def load_table(path):
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


def probability_fold_has_no_high_burden(n_total, n_high, n_fold, n_folds):
    """
    Under random allocation without replacement, the probability that a given fold of
    n_fold scans contains no high-burden scan, and that at least one of n_folds does.

    The second figure uses the union bound complement under independence, which is the
    calculation quoted in the reviewer response. It is approximate because folds are not
    independent, but the dependence is weak for these numbers.
    """
    p_one = hypergeom.pmf(0, n_total, n_high, n_fold)
    p_any = 1 - (1 - p_one) ** n_folds
    return p_one, p_any


def main():
    parser = argparse.ArgumentParser(description="Fold composition by WMH burden category")
    parser.add_argument("--in_path", required=True,
                        help="Table with one row per scan, containing fold and burden columns")
    parser.add_argument("--fold_col", default="Fold")
    parser.add_argument("--burden_col", default="Classification")
    parser.add_argument("--high_burden_label", default=None,
                        help="Value of the burden column denoting high burden, "
                             "for the exact calculation")
    parser.add_argument("--volume_col", default=None,
                        help="Optional reference volume column, for per-fold mean burden")
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = load_table(args.in_path)

    contingency = df.groupby([args.fold_col, args.burden_col]).size().unstack(fill_value=0)
    print("Burden category counts per fold:")
    print(contingency)
    contingency.to_csv(os.path.join(args.out_dir, "fold_burden_contingency.csv"))

    totals = df.groupby(args.burden_col).size()
    print("\nScans per burden category:")
    print(totals)
    totals.to_csv(os.path.join(args.out_dir, "burden_category_totals.csv"))

    chi2, p, dof, expected = chi2_contingency(contingency.values)
    print(f"\nChi-square test of fold x burden independence: "
          f"chi2 = {chi2:.4f}, dof = {dof}, p = {p:.4f}")
    print(f"Minimum expected cell count: {expected.min():.2f}")
    if expected.min() < 5:
        print("NOTE: an expected cell count is below 5, so the chi-square approximation "
              "is unreliable here. Quote the exact calculation below instead.")

    lines = [f"chi2 = {chi2:.6f}", f"dof = {dof}", f"p = {p:.6f}",
             f"min expected cell = {expected.min():.4f}"]

    if args.high_burden_label is not None:
        n_total = len(df)
        n_high = int((df[args.burden_col].astype(str) == str(args.high_burden_label)).sum())
        n_folds = df[args.fold_col].nunique()
        n_fold = int(round(n_total / n_folds))

        p_one, p_any = probability_fold_has_no_high_burden(n_total, n_high, n_fold, n_folds)
        msg = (f"\nExact calculation: {n_high} of {n_total} scans are high burden. "
               f"For a fold of {n_fold} scans under random allocation, "
               f"P(no high-burden scan) = {p_one:.4f}; "
               f"P(at least one of {n_folds} folds has none) = {p_any:.4f}.")
        print(msg)
        lines.append(msg.strip())

    with open(os.path.join(args.out_dir, "fold_burden_test.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")

    if args.volume_col:
        per_fold_volume = df.groupby(args.fold_col)[args.volume_col].agg(
            ['count', 'mean', 'median'])
        print("\nReference WMH volume per fold:")
        print(per_fold_volume)
        per_fold_volume.to_csv(os.path.join(args.out_dir, "fold_reference_volume.csv"))

    print("\nOutputs written to", args.out_dir)


if __name__ == "__main__":
    main()
