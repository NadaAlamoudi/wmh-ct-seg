#!/usr/bin/env bash
#
# The full pipeline, in order, from DICOM to the last statistical test.
# Reference for the command sequence; adapt the paths before running.
# See docs/INFERENCE.md to run the model on new scans instead.

set -euo pipefail

# --- Environment -------------------------------------------------------------
export WMH_DATA_ROOT=${WMH_DATA_ROOT:?set WMH_DATA_ROOT}
export WMH_RESULTS_ROOT=${WMH_RESULTS_ROOT:?set WMH_RESULTS_ROOT}
export FSL_DIR=${FSL_DIR:-/usr/local/fsl}
export NIFTYREG_DIR=${NIFTYREG_DIR:-/usr/local/niftyreg}
export DCM2NIIX=${DCM2NIIX:-$(command -v dcm2niix)}

RAW="$WMH_DATA_ROOT/MSS3"
WORK="$WMH_RESULTS_ROOT/mss3"
mkdir -p "$WORK"

# --- Stage 1: data preparation -----------------------------------------------
wmhct-convert-dicom --in_dir "$RAW/dicom" --dcm2niix "$DCM2NIIX"

wmhct-sort-bids --in_dir "$RAW/ct_nifti" --out_dir "$WORK/bids"

# Build the DICOM header table separately, then classify.
wmhct-classify-sequences \
  --in_csv "$WORK/dicom_headers.csv" \
  --out_csv "$WORK/dicom_headers_classified.csv"

wmhct-pair-labels \
  --mri_dir "$RAW/flair_nifti" \
  --wmh_dir "$RAW/wmh_masks_flair_space" \
  --out_dir "$WORK/paired"

# --- Stage 2: CT preprocessing -----------------------------------------------

# Step 1
wmhct-clip-hu --in_dir "$RAW/ct_nifti" --out_dir "$WORK/ct_clipped"

# Step 2
wmhct-skull-strip -in "$WORK/ct_clipped" -out "$WORK/ct_stripped"

# Step 3. Choose ONE windowing setup; the predefined window is the default condition.
wmhct-window-fixed --in_dir "$WORK/ct_stripped" --out_dir "$WORK/ct_windowed" \
  --window_center 40 --window_width 80 --qc_dir "$WORK/qc/windowing"

# --- Stage 3: registration ---------------------------------------------------
wmhct-register-flair2ct \
  --ct_dir "$WORK/ct_windowed" \
  --mri_dir "$RAW/flair_nifti" \
  --mask_dir "$RAW/wmh_masks_flair_space" \
  --output_dir "$WORK/registration" \
  --trans_masks_dir "$WORK/masks_ct_space" \
  --tool flirt \
  --id_pattern '(\d+)_'

# --- Stage 4: nnU-Net --------------------------------------------------------
# Arrange $WORK/nnunet_task/{imagesTr,labelsTr} in nnU-Net naming first.
wmhct-dataset-json --task_dir "$WORK/nnunet_task" --dataset_name Dataset0XX_MSS3_CT --modality CT
# Then follow docs/TRAINING.md.

# --- Stage 5: back projection ------------------------------------------------
# Predictions come out of nnU-Net already in native CT space, so the only step here is
# the inverse transform back into native FLAIR space.
wmhct-inverse-to-flair \
  --nnunet_ct_dir "$WORK/predictions_ct_space" \
  --flair_dir "$RAW/flair_nifti" \
  --flair_gt_dir "$RAW/wmh_masks_flair_space" \
  --trans_matrix_dir "$WORK/registration" \
  --output_dir "$WORK/predictions_flair_space" \
  --csv_file "$WMH_RESULTS_ROOT/tables/per_case_flirt.csv" \
  --tool flirt

# --- Stage 6a: per-case metrics for every experiment -----------
METRICS="$WMH_RESULTS_ROOT/tables/experiment_metrics"

wmhct-per-case-metrics \
  --results_dir "$WMH_RESULTS_ROOT/nnUNet_results" \
  --gt_base_dir "$WMH_RESULTS_ROOT/nnUNet_preprocessed" \
  --output_dir "$METRICS"

wmhct-summarise-experiments --in_dir "$METRICS"

# Model 7, the final model, is Dataset051. Used by wmhct-stroke-dice below.
M7="$METRICS/Dataset051_retraining_iDB_MSS3_IST3_metrics.xlsx"

# --- Stage 6b: volume agreement ----------------------------------------------
wmhct-volumes \
  --ct_pred_dir "$WORK/predictions_ct_space" \
  --ct_gt_dir "$WORK/masks_ct_space" \
  --flair_gt_dir "$RAW/wmh_masks_flair_space" \
  --flair_pred_dir "$WORK/predictions_flair_space" \
  --out_csv "$WMH_RESULTS_ROOT/tables/WMH_volumes_model7.csv"

wmhct-figures-10-11 \
  --in_csv "$WMH_RESULTS_ROOT/tables/WMH_volumes_model7.csv" \
  --out_dir "$WMH_RESULTS_ROOT/figures"

wmhct-volume-ccc --in_csv "$WMH_RESULTS_ROOT/tables/WMH_volumes_model7.csv" \
  --out_dir "$WMH_RESULTS_ROOT/figures/volume_correlation" --ccc_ddof 1

# --- Sections 3.5 and 3.6 - modality, inter-scan interval and WMH change -----
wmhct-wmh-progression \
  --participants_xlsx "$WMH_RESULTS_ROOT/tables/patients_with_CT_at_V0.xlsx" \
  --v0_mask_dir "$WORK/predictions_flair_space" \
  --v1_mask_dir "$RAW/WMHmasks_from_flair_v1" \
  --out_mask_dir "$WMH_RESULTS_ROOT/wmh_progression_V0_to_V1" \
  --out_xlsx "$WMH_RESULTS_ROOT/tables/WMH_progression_results_at_V1_CT_MRI.xlsx"

wmhct-modality-comparison \
  --combined_xlsx "$WMH_RESULTS_ROOT/tables/combined_data_for_analysis_allData.xlsx" \
  --out_dir "$WMH_RESULTS_ROOT/tables/section_3_5"

wmhct-change-regression \
  --combined_xlsx "$WMH_RESULTS_ROOT/tables/combined_data_for_analysis_allData.xlsx" \
  --out_dir "$WMH_RESULTS_ROOT/tables/section_3_6"

# --- Section 3.7 - stroke lesions --------------------------------------------
# Needs one --root holding the reference, predicted and stroke masks in FLAIR space.
STROKE_ROOT="$WORK/stroke_overlap"

wmhct-stroke-overlap --root "$STROKE_ROOT"

# The regression models. Pass --model7_metrics so the Dice column is re-joined from the
# verified Model 7 workbook; without it the module refuses to run. See the README.

wmhct-stroke-dice \
  --stroke_xlsx "$WMH_RESULTS_ROOT/tables/stroke_info_recomputed.xlsx" \
  --model7_metrics "$M7" \
  --out_dir "$WMH_RESULTS_ROOT/tables/section_3_7"

# --- Section 3.8 - perivascular spaces ---------------------------------------
PVS="$WMH_RESULTS_ROOT/pvs"

wmhct-pvs-density        --roi_manifest "$PVS/PVS_ROI/roi_manifest_V1.csv" \
                         --spreadsheet_xlsx "$WMH_RESULTS_ROOT/tables/MValdesHernandez_PVS_Fazekas.xlsx" \
                         --out_csv "$PVS/pvs_density_v1.csv"
wmhct-wmh-metrics-by-roi --manifest "$PVS/roi_wmh_manifest_V1.csv" \
                         --out_csv "$PVS/wmh_metrics_by_roi_V1.csv"
wmhct-pvs-associations   --pvs_csv "$PVS/pvs_density_v1.csv" \
                         --wmh_metrics_csv "$PVS/wmh_metrics_by_roi_V1.csv" \
                         --out_dir "$PVS/associations"

echo "Pipeline complete."
