# Data files

Paste here the files the analysis code needs. See the Data and model availability section
of the README for how to obtain the MSS3 and IST-3 data.

## Tables

| File | Needed by |
|---|---|
| `Dataset034_..._metrics.xlsx` (Model 5) | `wmhct-burden-class-figures`, `wmhct-model5-vs-model7` |
| `Dataset050_..._metrics.xlsx` (Model 6) | `wmhct-burden-class-figures` |
| `Dataset051_..._metrics.xlsx` (Model 7) | `wmhct-burden-class-figures`, `wmhct-model5-vs-model7`, `wmhct-stroke-dice` (`--model7_metrics`), `wmhct-build-release-table` |
| folder of per-experiment metrics workbooks | `wmhct-summarise-experiments` |
| `wmh_volume_classification.xlsx` | `wmhct-fold-composition` |
| `WMH_volumes_..._FINAL.csv` | `wmhct-figures-10-11`, `wmhct-volume-ccc` |
| `combined_data_for_analysis_allData.xlsx` | `wmhct-modality-comparison`, `wmhct-change-regression` |
| `updated_results_with_dice_volumes_stroke_info.xlsx` | `wmhct-stroke-dice` |
| `wmh_stroke_overlap_per_subject.csv` | `wmhct-lesion-excluded` |
| `patients_with_CT_at_V0.xlsx` | `wmhct-wmh-progression` |
| `roi_manifest_V1.csv` | `wmhct-pvs-density` |
| `roi_wmh_manifest_V1.csv` | `wmhct-wmh-metrics-by-roi` |
| PVS and Fazekas spreadsheet | `wmhct-pvs-density` |
| `pvs_density_v1.csv`, `wmh_metrics_by_roi_V1.csv` | `wmhct-pvs-associations` |

Sections 3.5 and 3.6 read the sheet `For_statistical_analyses`; pass it with `--sheet`.

## Imaging

| Folder | Needed by |
|---|---|
| CT and FLAIR NIfTI, reference WMH masks | the preprocessing and registration stages |
| predictions and reference masks in CT space | `wmhct-per-case-metrics` |
| predictions and reference masks in FLAIR space | `wmhct-volumes`, `wmhct-wmh-progression` |
| reference, predicted and stroke masks in FLAIR space, in one folder | `wmhct-stroke-overlap` (`--root`) |
| basal ganglia and centrum semiovale ROI masks | `wmhct-wmh-metrics-by-roi` |

`wmhct-rename-masks` gives the three mask sets one naming convention before the `--root`
commands are run. It is a dry run unless `--apply` is given, and `--undo` reverses it.
