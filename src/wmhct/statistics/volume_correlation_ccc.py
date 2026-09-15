"""
Supporting analysis - Pearson and Lin's concordance correlation of WMH volume.
Run: wmhct-volume-ccc --help
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

from wmhct import config  # noqa: E402


def concordance_correlation_coefficient(y_true, y_pred, ddof=0):
    """Lin's CCC. Set ddof=1 to use sample estimators throughout."""
    mean_true, mean_pred = np.mean(y_true), np.mean(y_pred)
    var_true = np.var(y_true, ddof=ddof)
    var_pred = np.var(y_pred, ddof=ddof)
    covariance = np.cov(y_true, y_pred, ddof=ddof)[0, 1]
    return (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2)


def format_p(p_value):
    if p_value < 1e-5:
        return "<0.00001"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.4f}"


def get_regression_eq(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    return f"y = {slope:.2f}x + {intercept:.2f}"


def scatter_panel(x, y, xlabel, ylabel, title, out_path):
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x, y=y)
    plt.text(0.05, 0.95, get_regression_eq(x, y), transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top')
    plt.title(title, fontsize=15)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Volume correlation and CCC analysis")
    parser.add_argument("--in_csv", default=str(config.VOLUME_TABLE))
    parser.add_argument("--out_dir", default=str(config.FIGURES / "volume_correlation"))
    parser.add_argument("--ccc_ddof", type=int, default=0, choices=[0, 1],
                        help="0 reproduces the original; 1 uses sample estimators throughout")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    results_df = pd.read_csv(args.in_csv, dtype={'Patient ID': str}).dropna()

    comparisons = [
        ('nnUnet CT-based Volume (ml)', 'Registered to CT_space GT Volume (ml)',
         'CT space: predicted vs reference (CT-registered)'),
        ('nnUnet CT-based Volume (ml)', 'FLAIR GT Volume (ml)',
         'Predicted CT-space volume vs reference FLAIR volume'),
        ('FLAIR GT Volume (ml)', 'nnUNet Volume in FLAIR_space (ml)',
         'FLAIR space: reference vs predicted'),
    ]

    for n, (xcol, ycol, label) in enumerate(comparisons, start=1):
        x, y = results_df[xcol], results_df[ycol]
        r, p = pearsonr(x, y)
        ccc = concordance_correlation_coefficient(x, y, ddof=args.ccc_ddof)
        print(f"[{n}] {label}: r = {r:.4f} (p = {format_p(p)}), "
              f"CCC = {ccc:.4f} (ddof={args.ccc_ddof}), n = {len(x)}")

        scatter_panel(x, y, xcol, ycol,
                      f"{label}\nr = {r:.4f}, p = {format_p(p)}, CCC = {ccc:.4f}",
                      os.path.join(args.out_dir, f"comparison_{n}.png"))

    # Residuals and outliers, FLAIR space
    results_df['Residuals'] = (results_df['FLAIR GT Volume (ml)']
                               - results_df['nnUNet Volume in FLAIR_space (ml)'])
    mean_residual = results_df['Residuals'].mean()
    std_residual = results_df['Residuals'].std()
    outlier_threshold = 3 * std_residual
    outliers = results_df[np.abs(results_df['Residuals']) > outlier_threshold]

    print(f"\nMean residual: {mean_residual:.4f} mL, SD: {std_residual:.4f} mL, "
          f"outlier threshold (3 SD): {outlier_threshold:.4f} mL")
    print(f"Outliers ({len(outliers)}):")
    print(outliers[['Patient ID', 'Residuals']])
    outliers.to_csv(os.path.join(args.out_dir, 'outliers_WMH_volume_comparison.csv'), index=False)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=results_df['nnUNet Volume in FLAIR_space (ml)'],
                    y=results_df['Residuals'])
    plt.axhline(0, color='red', linestyle='--')
    plt.title('Residuals vs predicted WMH volume (FLAIR space)', fontsize=15)
    plt.xlabel('Predicted WMH volume (mL), FLAIR space', fontsize=14)
    plt.ylabel('Residual (mL): reference - predicted', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'residuals_vs_predicted.png'), dpi=200)
    plt.close()

    print("\nOutputs written to", args.out_dir)


if __name__ == "__main__":
    main()
