# Statistics index

Every statistical test reported in the paper, with the command that produces it.

Measurement code — anything that turns NIfTI masks into numbers — lives in
`wmhct.evaluation`. Anything that computes a p-value, a correlation, an effect size or
limits of agreement lives in `wmhct.statistics`.

Collected data files go in `data/`. See `data/MANIFEST.md` for what
each module needs.

---

## Index of every statistical test in the paper

### A. Segmentation performance

| Module | Test | Reported as |
|---|---|---|
| `wmhct-burden-class-figures` | Paired t-test (or Wilcoxon) on Dice and MAE, Models 5/6/7 within each burden stratum. Cohen's d. Nine comparisons. | Dice and MAE by WMH volume class figures |
| `wmhct-model5-vs-model7` | Paired test on per-scan Model 7 minus Model 5 differences, overall and within strata | Difference panels |
| `wmhct-fold-composition` | Chi-square of fold by burden category, plus the exact hypergeometric probability that a fold contains no high-burden scan | Figure 5, Supplementary Table S2, response to R1.4(2) |

### B. Volume agreement

| Module | Test | Reported as |
|---|---|---|
| `wmhct-figures-10-11` | Pearson correlation and Deming (orthogonal) regression; Bland-Altman with fixed and regression-based limits | Figures 10 and 11 |
| `wmhct-volume-ccc` | Pearson and Lin's concordance correlation; residual and outlier inspection | supporting analysis |

### D. Scanning modality, time interval and stroke lesions

| Module | Section | Test |
|---|---|---|
| `wmhct-modality-comparison` | 3.5 | Shapiro-Wilk and Levene, then t-test, Welch or Mann-Whitney per outcome, CT-MRI vs MRI-MRI, on six WMH change outcomes, with effect sizes |
| `wmhct-change-regression` | 3.6 | OLS of each outcome on time interval, modality, age and sex, with residual diagnostics |
| `wmhct-stroke-dice` | 3.7 | OLS of Dice on stroke lesion volume, stroke type and WMH volume, for old and index lesions; Pearson and Spearman correlations |

### E. PVS and Fazekas (Section 3.8)

| Module | Test | Reported as |
|---|---|---|
| `wmhct-pvs-associations` | Spearman of PVS density against false-positive burden (family A); Spearman of reader-assigned Fazekas subscales against WMH burden (family B); optional median-split Mann-Whitney | PVS association analysis and the Fazekas correlation |

Upstream, no inference: `wmhct-pvs-density`, `wmhct-wmh-metrics-by-roi`, and the
the ROI manifest, which is prepared upstream from the study tree.

There is no interobserver analysis in the revised manuscript, so no module for it is
included here.

---

## Running order

```
roi_manifest_V1.csv + the curated PVS spreadsheet   (prepared upstream)
        |
wmhct-pvs-density           spreadsheet + manifest -> pvs_density_v1.csv
wmhct-wmh-metrics-by-roi    masks + manifest       -> wmh_metrics_by_roi_V1.csv
        |
wmhct-pvs-associations      both of the above      -> association_stats_summary.csv
```

The segmentation-performance and volume-agreement modules are independent of the PVS
chain; they consume the per-case metrics workbooks and the volume table produced by
`wmhct-per-case-metrics` and `wmhct-volumes`.

### F. Section 3.7, stroke lesions

| Module | Test | Reported as |
|---|---|---|
| `wmhct-stroke-dice` | OLS of Dice on lesion volume, lesion presence and subtype, adjusted for reference WMH volume; joint F-tests on the subtype contrasts; standardised betas on each model's own sample | Section 3.7, Table S4 |
| `wmhct-stroke-overlap` | Direct voxelwise intersection of predicted WMH with the lesion; bootstrap CIs; Dice and volume difference recomputed with the lesion territory removed | Section 3.7, Figure 14 |
| `wmhct-lesion-excluded` | Paired models on Dice with and without the lesion territory | the -0.44 to -0.36 comparison in Section 3.7 |

`wmhct-stroke-overlap` needs one `--root` holding the four mask folders.
Run `wmhct-rename-masks --apply` first.

---

