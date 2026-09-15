"""
Figures 10 and 11 - volume agreement between predicted and reference WMH volume.

Run: wmhct-figures-10-11 --help

Note: The Bland-Altman difference is predicted - reference, so a negative bias is underestimation.
"""

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd
from scipy import odr
from scipy.stats import pearsonr, linregress, spearmanr

from wmhct import config

matplotlib.use(os.environ.get("MPLBACKEND", "Agg"))
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

CT_GT_COL = 'Registered to CT_space GT Volume (ml)'
CT_PRED_COL = 'nnUnet CT-based Volume (ml)'
FLAIR_GT_COL = 'FLAIR GT Volume (ml)'
FLAIR_PRED_COL = 'nnUNet Volume in FLAIR_space (ml)'


def deming_regression(x, y, lambda_ratio=1.0):
    """Orthogonal-distance regression, accounting for error in both variables."""
    def linear_model(params, t):
        return params[0] * t + params[1]

    data = odr.RealData(x, y, sx=np.ones_like(x),
                        sy=np.ones_like(y) * np.sqrt(lambda_ratio))
    fit = odr.ODR(data, odr.Model(linear_model), beta0=[1.0, 0.0]).run()
    slope, intercept = fit.beta
    slope_se, intercept_se = fit.sd_beta
    return slope, intercept, slope_se, intercept_se


def format_p(p_value):
    if p_value < 1e-5:
        return "<0.00001"
    if p_value < 0.001:
        return "<0.001"
    return f"{p_value:.4f}"


def confidence_interval(estimate, standard_error):
    return estimate - 1.96 * standard_error, estimate + 1.96 * standard_error


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_csv", default=str(config.VOLUME_TABLE),
                    help="volume table written by wmhct-volumes")
    ap.add_argument("--out_dir", default=str(config.FIGURES))
    ap.add_argument("--lambda_ratio", type=float, default=1.0,
                    help="assumed ratio of error variances for the Deming fit; the "
                         "manuscript reports sensitivity at 0.5 and 2.0")
    ap.add_argument("--show", action="store_true", help="also open the figure windows")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log = []

    def say(line=""):
        print(line)
        log.append(line)

    results_df = pd.read_csv(args.in_csv).dropna()
    say(f"Loaded {len(results_df)} scans from {args.in_csv}")

    ct_gt = results_df[CT_GT_COL]
    ct_pred = results_df[CT_PRED_COL]
    flair_gt = results_df[FLAIR_GT_COL]
    flair_pred = results_df[FLAIR_PRED_COL]

    # ------------------------------------------------------------------
    # Figure 10: predicted against reference, in both spaces.
    # ------------------------------------------------------------------
    comparisons = [
        (ct_gt, ct_pred, 'CT space',
         'Reference WMH volume in CT space (ml)',
         'Predicted WMH volume in CT space (ml)'),
        (flair_gt, flair_pred, 'Native FLAIR space',
         'Reference WMH volume in FLAIR space (ml)',
         'Predicted WMH volume in FLAIR space (ml)'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    for ax, panel, (x, y, space, x_label, y_label) in zip(axes, ['a', 'b'], comparisons):
        r, p = pearsonr(x, y)
        slope, intercept, slope_se, intercept_se = deming_regression(
            x, y, lambda_ratio=args.lambda_ratio)
        slope_ci = confidence_interval(slope, slope_se)
        intercept_ci = confidence_interval(intercept, intercept_se)

        say()
        say(f"({panel}) {space}")
        say(f"  Pearson r = {r:.4f}, p = {format_p(p)}")
        say(f"  Deming slope = {slope:.4f} "
            f"(95% CI {slope_ci[0]:.4f} to {slope_ci[1]:.4f})")
        say(f"  Deming intercept = {intercept:.4f} ml "
            f"(95% CI {intercept_ci[0]:.4f} to {intercept_ci[1]:.4f})")

        axis_max = max(x.max(), y.max()) * 1.06
        fit_line = np.linspace(0, axis_max, 100)

        ax.plot([0, axis_max], [0, axis_max], color='grey', linestyle='--',
                linewidth=1.2, label='Identity line')
        ax.plot(fit_line, slope * fit_line + intercept, color='firebrick',
                linewidth=1.8, label=f'Deming: y = {slope:.3f}x {intercept:+.3f}')
        sns.scatterplot(x=x, y=y, ax=ax, s=45, color='steelblue',
                        edgecolor='white', linewidth=0.6, alpha=0.9, legend=False)

        ax.text(0.04, 0.96, f'Pearson r = {r:.3f}\nn = {len(x)}',
                transform=ax.transAxes, fontsize=13, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='lightgrey'))
        ax.set_title(f'({panel}) {space}', fontsize=16)
        ax.set_xlabel(x_label, fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.set_xlim(0, axis_max)
        ax.set_ylim(0, axis_max)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=12)
        ax.legend(loc='lower right', fontsize=11)
        sns.despine(ax=ax)
        ax.grid(alpha=0.2)

    plt.tight_layout()
    fig10_png = os.path.join(args.out_dir, 'Figure10_volume_correlation.png')
    plt.savefig(fig10_png, dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(args.out_dir, 'Figure10_volume_correlation.pdf'),
                bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------
    # Bland-Altman, native FLAIR space. difference = predicted - reference.
    # ------------------------------------------------------------------
    ba = results_df.copy()
    ba['MeanVolume'] = (flair_gt + flair_pred) / 2
    ba['Difference'] = flair_pred - flair_gt

    mean_diff = ba['Difference'].mean()
    std_diff = ba['Difference'].std()
    upper_limit = mean_diff + 1.96 * std_diff
    lower_limit = mean_diff - 1.96 * std_diff
    under_estimated = (ba['Difference'] < 0).mean() * 100

    say()
    say(f"Bland-Altman (predicted - reference): bias = {mean_diff:.2f} ml, "
        f"95% limits of agreement = {lower_limit:.2f} to {upper_limit:.2f} ml")
    say(f"Mean reference volume = {flair_gt.mean():.2f} ml, "
        f"mean predicted volume = {flair_pred.mean():.2f} ml")
    say(f"Volume underestimated in {under_estimated:.1f}% of scans")

    trend = linregress(ba['MeanVolume'], ba['Difference'])
    rho, rho_p = spearmanr(ba['MeanVolume'], ba['Difference'].abs())
    residual_sd = np.std(ba['Difference']
                         - (trend.intercept + trend.slope * ba['MeanVolume']), ddof=2)

    say(f"Heteroscedasticity: difference vs mean slope = {trend.slope:.4f}, "
        f"p = {format_p(trend.pvalue)}")
    say(f"                    |difference| vs mean rho = {rho:.3f}, "
        f"p = {format_p(rho_p)}")
    say(f"Regression-based limits: difference = {trend.intercept:.3f} + "
        f"{trend.slope:.4f} * mean +/- {1.96 * residual_sd:.3f} ml")

    say()
    say("Bias by WMH burden:")
    burden_groups = [('Low (<=10 ml)', flair_gt <= 10),
                     ('Medium (10-25 ml)', (flair_gt > 10) & (flair_gt <= 25)),
                     ('High (>25 ml)', flair_gt > 25)]
    for label, mask in burden_groups:
        group_diff = ba['Difference'][mask]
        say(f"  {label:20s} n = {mask.sum():3d}, bias = {group_diff.mean():+.2f} ml "
            f"({100 * group_diff.mean() / flair_gt[mask].mean():+.1f}% of reference)")

    # ------------------------------------------------------------------
    # Figure 11
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(10, 6.5))
    mean_axis = np.linspace(0, ba['MeanVolume'].max() * 1.06, 200)
    trend_line = trend.intercept + trend.slope * mean_axis

    plt.fill_between(mean_axis, trend_line - 1.96 * residual_sd,
                     trend_line + 1.96 * residual_sd,
                     color='orange', alpha=0.13, label='Regression-based 95% LoA')
    plt.plot(mean_axis, trend_line, color='orange', linestyle='-.', linewidth=1.5,
             label=f'Proportional-bias trend: slope = {trend.slope:.3f}, '
                   f'p = {format_p(trend.pvalue)}')
    plt.axhline(mean_diff, color='firebrick', linewidth=1.8,
                label=f"Mean bias = {mean_diff:+.2f} ml")
    plt.axhline(upper_limit, color='dimgrey', linestyle='--', linewidth=1.3,
                label=f"Constant 95% LoA = {lower_limit:+.2f} to {upper_limit:+.2f} ml")
    plt.axhline(lower_limit, color='dimgrey', linestyle='--', linewidth=1.3)
    plt.axhline(0, color='grey', linestyle=':', linewidth=1)
    plt.scatter(ba['MeanVolume'], ba['Difference'], s=45, color='steelblue',
                edgecolor='white', linewidth=0.6, alpha=0.9)

    plt.title('Bland-Altman: reference vs automated WMH volume [FLAIR space] '
              f'(n = {len(ba)})', fontsize=16)
    plt.xlabel('Mean of reference and predicted WMH volume (ml)', fontsize=14)
    plt.ylabel('Predicted - reference WMH volume (ml)', fontsize=14)
    plt.text(0.98, 0.03, 'Positive difference = overestimation by the model',
             transform=plt.gca().transAxes, fontsize=12, style='italic',
             color='dimgrey', horizontalalignment='right')
    plt.xlim(0, ba['MeanVolume'].max() * 1.06)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(loc='upper left', fontsize=11)
    sns.despine()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'Figure11_bland_altman.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(args.out_dir, 'Figure11_bland_altman.pdf'),
                bbox_inches='tight')
    if args.show:
        plt.show()
    plt.close(fig)

    # ------------------------------------------------------------------
    # Scans furthest from the mean difference, kept for inspection.
    # ------------------------------------------------------------------
    outlier_threshold = 3 * std_diff
    outliers = ba[np.abs(ba['Difference'] - mean_diff) > outlier_threshold]
    say()
    say(f"Outlier threshold (3 SD of the differences) = {outlier_threshold:.2f} ml")
    if len(outliers) > 0:
        say(outliers[['Patient ID', FLAIR_GT_COL, FLAIR_PRED_COL,
                      'Difference']].to_string(index=False))
        outliers.to_csv(os.path.join(args.out_dir,
                                     'outliers_WMH_volume_comparison.csv'), index=False)
    else:
        say("No scan exceeded the threshold")

    values_path = os.path.join(args.out_dir, 'figures_10_11_values.txt')
    with open(values_path, 'w') as handle:
        handle.write("\n".join(log) + "\n")
    print(f"\nFigures and numeric summary written to {args.out_dir}")


if __name__ == "__main__":
    main()
