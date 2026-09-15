"""
Central path and tool configuration.
"""

import os
from pathlib import Path


def _env_path(name: str, default: str) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


# --------------------------------------------------------------------------
# Roots
# --------------------------------------------------------------------------
DATA_ROOT = _env_path("WMH_DATA_ROOT", "./data")
RESULTS_ROOT = _env_path("WMH_RESULTS_ROOT", "./results")

# --------------------------------------------------------------------------
# External tools
# --------------------------------------------------------------------------
FSL_DIR = _env_path("FSL_DIR", "/usr/local/fsl")
FLIRT = str(FSL_DIR / "bin" / "flirt")
CONVERT_XFM = str(FSL_DIR / "bin" / "convert_xfm")

NIFTYREG_DIR = _env_path("NIFTYREG_DIR", "/usr/local/niftyreg")
REG_ALADIN = str(NIFTYREG_DIR / "bin" / "reg_aladin")
REG_RESAMPLE = str(NIFTYREG_DIR / "bin" / "reg_resample")
REG_TRANSFORM = str(NIFTYREG_DIR / "bin" / "reg_transform")

DCM2NIIX = os.environ.get("DCM2NIIX", "dcm2niix")

# Template used for CT to template-space registration (template-space experiment).
CT_TEMPLATE = _env_path("CT_TEMPLATE", "./templates/ct_template.nii.gz")

# --------------------------------------------------------------------------
# Cohort sub-trees.
# --------------------------------------------------------------------------
MSS3_DICOM = DATA_ROOT / "MSS3" / "dicom"
MSS3_CT = DATA_ROOT / "MSS3" / "ct_nifti"
MSS3_FLAIR = DATA_ROOT / "MSS3" / "flair_nifti"
MSS3_WMH_MASKS = DATA_ROOT / "MSS3" / "wmh_masks_flair_space"

IST3_CT = DATA_ROOT / "IST3" / "ct_nifti"
IST3_FLAIR = DATA_ROOT / "IST3" / "flair_nifti"
IST3_WMH_MASKS = DATA_ROOT / "IST3" / "wmh_masks_flair_space"

CIM_CT = DATA_ROOT / "CIM" / "ct_nifti"
CIM_WMH_MASKS = DATA_ROOT / "CIM" / "wmh_masks"

# --------------------------------------------------------------------------
# nnU-Net
# --------------------------------------------------------------------------
NNUNET_RAW = RESULTS_ROOT / "nnUNet_raw"
NNUNET_PREPROCESSED = RESULTS_ROOT / "nnUNet_preprocessed"
NNUNET_RESULTS = RESULTS_ROOT / "nnUNet_results"

# Final model (Model 7) prediction directories.
MODEL7_PREDICTIONS_CT_SPACE = NNUNET_RESULTS / "model7" / "predictions_ct_space"
MODEL7_PREDICTIONS_FLAIR_SPACE = NNUNET_RESULTS / "model7" / "predictions_flair_space"

# --------------------------------------------------------------------------
# Tabular and figure outputs
# --------------------------------------------------------------------------
TABLES = RESULTS_ROOT / "tables"
FIGURES = RESULTS_ROOT / "figures"

VOLUME_TABLE = TABLES / "WMH_volumes_exp51_model7_FINAL.csv"

# Per-case metrics for the final model. Written by `wmhct-per-case-metrics` as an Excel
# workbook, one per experiment, with the per-scan rows on the `All_Folds` sheet.
# Model 5 = Dataset034, Model 6 = Dataset050, Model 7 = Dataset051.
EXPERIMENT_METRICS_DIR = TABLES / "experiment_metrics"
PER_CASE_METRICS = EXPERIMENT_METRICS_DIR / "Dataset051_retraining_iDB_MSS3_IST3_metrics.xlsx"

MODEL5_METRICS = EXPERIMENT_METRICS_DIR / "Dataset034_MSS3_only_CT_predefined_window_cropped_affine_metrics.xlsx"
MODEL6_METRICS = EXPERIMENT_METRICS_DIR / "Dataset050_retraining_iDB_MSS3_metrics.xlsx"
MODEL7_METRICS = PER_CASE_METRICS

# --------------------------------------------------------------------------
# Reference masks and stroke lesions in native FLAIR space
#
# These four folders are what the Section 3.7 overlap chain reads.
# The overlap and figure modules take a single --root holding all
# of them, so these entries are for the modules that address them individually.
# --------------------------------------------------------------------------
WMH_REFERENCE_FLAIR_SPACE = DATA_ROOT / "GT_WMHmasks_MSS3_in_FLAIR_space"
STROKE_MASKS_FLAIR_SPACE = DATA_ROOT / "Stroke_masks_in_FLAIR_space"
FLAIR_MSS3 = DATA_ROOT / "FLAIR_MSS3"
OVERLAP_ROOT = DATA_ROOT  # --root for wmh-stroke-overlap

# --------------------------------------------------------------------------
# Section 3.5 and 3.6: WMH change between visits
# --------------------------------------------------------------------------
# Participants with a CT at the acute presentation visit (V0). First column is
# the participant ID.
CT_AT_V0_LIST = TABLES / "patients_with_CT_at_V0.xlsx"

# Reference WMH masks derived from the V1 research FLAIR, named <digits>_gt.nii.gz.
WMH_MASKS_FLAIR_V1 = DATA_ROOT / "WMHmasks_from_flair_v1"

# Voxelwise stable / growth / regression masks and the table derived from them.
WMH_PROGRESSION_MASKS = RESULTS_ROOT / "wmh_progression_V0_to_V1"
WMH_PROGRESSION_TABLE = TABLES / "WMH_progression_results_at_V1_CT_MRI.xlsx"

# The MRI-MRI group, produced in the primary study by the same voxelwise definitions.
WMH_PROGRESSION_TABLE_MRI_MRI = TABLES / "WMH_progression_at_V1_MRI_MRI.xlsx"

# Combined analysis table, before and after the visit dates are joined on.
WMH_CHANGE_TABLE = TABLES / "combined_data_for_analysis.xlsx"
WMH_CHANGE_TABLE_WITH_DATES = TABLES / "combined_data_for_analysis_allData.xlsx"

# MSS3 scan dates and demographics.
MSS3_SCAN_DATES = TABLES / "MSS3_scan_dates_up_to_1yr_followup.xlsx"

# --------------------------------------------------------------------------
# Section 3.7: stroke lesion effect on segmentation agreement
# --------------------------------------------------------------------------
# Per-scan Dice, WMH volume and stroke lesion volumes. Pass --model7_metrics to
STROKE_INFO_TABLE = TABLES / "updated_results_with_dice_volumes_stroke_info.xlsx"

# --------------------------------------------------------------------------
# Section 3.8: perivascular spaces
# --------------------------------------------------------------------------
PVS_ROOT = DATA_ROOT / "PVS"
PVS_ROI_ROOT = PVS_ROOT / "PVS_ROI"        # left_BG/ right_BG/ left_CSO/ right_CSO/
ROI_INVENTORY = PVS_ROI_ROOT / "roi_inventory.csv"
ROI_MANIFEST_V1 = PVS_ROI_ROOT / "roi_manifest_V1.csv"
ROI_WMH_MANIFEST_V1 = PVS_ROOT / "roi_wmh_manifest_V1.csv"

# Curated PVS and Fazekas spreadsheet. V1_PVLOverall and V1_DWMLOverall are the
PVS_SPREADSHEET = TABLES / "MValdesHernandez_PVS_Fazekas.xlsx"

PVS_DENSITY_TABLE = TABLES / "pvs_density_v1.csv"
WMH_METRICS_BY_ROI = TABLES / "wmh_metrics_by_roi_V1.csv"
PVS_MERGED_TABLE = TABLES / "merged_pvs_wmh_V1.csv"


def ensure_output_dirs():
    """Create the standard output directories if they do not exist."""
    for d in (RESULTS_ROOT, TABLES, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
