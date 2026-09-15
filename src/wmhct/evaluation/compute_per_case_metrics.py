"""
Stage 6.0 - Per-case segmentation metrics for every experiment and fold.

Manuscript: upstream source for Table V and the burden-class figures
Run: wmhct-per-case-metrics --help
"""

import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label
from scipy.spatial.distance import directed_hausdorff


def calculate_volume(mask_data, img):
    """Lesion volume in mm3, from the voxel count and the header zooms."""
    lesion_voxels = np.sum(mask_data == 1)
    voxel_volume = np.prod(img.header.get_zooms())
    return lesion_voxels * voxel_volume


def calculate_dice(gt_data, pred_data):
    intersection = np.sum((gt_data == 1) & (pred_data == 1))
    gt_sum = np.sum(gt_data == 1)
    pred_sum = np.sum(pred_data == 1)
    return (2 * intersection) / (gt_sum + pred_sum) if (gt_sum + pred_sum) != 0 else 0


def calculate_sensitivity(gt_data, pred_data):
    true_positive = np.sum((gt_data == 1) & (pred_data == 1))
    false_negative = np.sum((gt_data == 1) & (pred_data == 0))
    denom = true_positive + false_negative
    return true_positive / denom if denom != 0 else 0


def calculate_precision(gt_data, pred_data):
    true_positive = np.sum((gt_data == 1) & (pred_data == 1))
    false_positive = np.sum((gt_data == 0) & (pred_data == 1))
    denom = true_positive + false_positive
    return true_positive / denom if denom != 0 else 0


def calculate_ldr(gt_data, pred_data):
    """Lesion detection rate: reference components overlapping any predicted voxel."""
    gt_lesions, gt_num = label(gt_data)
    pred_lesions, _ = label(pred_data)
    detected = sum(1 for i in range(1, gt_num + 1) if np.any(pred_lesions[gt_lesions == i]))
    return detected / gt_num if gt_num != 0 else 0


def calculate_fpr(gt_data, pred_data):
    """False positive rate over all background voxels. See the module docstring."""
    false_positive = np.sum((gt_data == 0) & (pred_data == 1))
    true_negative = np.sum((gt_data == 0) & (pred_data == 0))
    denom = false_positive + true_negative
    return false_positive / denom if denom != 0 else 0


def calculate_boundary_metrics(gt_data, pred_data):
    """See the HD95 caveat in the module docstring: this is not a surface-distance HD95."""
    gt_points = np.argwhere(gt_data == 1)
    pred_points = np.argwhere(pred_data == 1)
    if gt_points.size == 0 or pred_points.size == 0:
        return None
    return np.percentile([
        directed_hausdorff(gt_points, pred_points)[0],
        directed_hausdorff(pred_points, gt_points)[0],
    ], 95)


def process_experiment(experiment_path, gt_dir, n_folds=5):
    """Return a list of per-case metric rows for one experiment."""
    rows = []
    mismatches = []

    for trainer in sorted(os.listdir(experiment_path)):
        trainer_path = os.path.join(experiment_path, trainer)
        if not os.path.isdir(trainer_path):
            continue

        print(f"  Trainer: {trainer}")

        for fold in range(n_folds):
            fold_dir = os.path.join(trainer_path, f"fold_{fold}", "validation")
            if not os.path.exists(fold_dir):
                print(f"    fold {fold} not found")
                continue

            for file in sorted(os.listdir(fold_dir)):
                if not file.endswith(".nii.gz"):
                    continue

                study_id = file.split(".nii.gz")[0]
                pred_path = os.path.join(fold_dir, file)
                gt_path = os.path.join(gt_dir, f"{study_id}.nii.gz")

                try:
                    if not os.path.exists(gt_path):
                        print(f"    no GT for {study_id} (fold {fold})")
                        continue

                    gt_img = nib.load(gt_path)
                    pred_img = nib.load(pred_path)

                    if gt_img.shape != pred_img.shape:
                        mismatches.append((study_id, fold, gt_img.shape, pred_img.shape))
                        continue

                    gt_data = gt_img.get_fdata().astype(bool)
                    pred_data = pred_img.get_fdata().astype(bool)

                    gt_volume = calculate_volume(gt_data, gt_img) / 1000    # mL
                    pred_volume = calculate_volume(pred_data, pred_img) / 1000

                    rows.append({
                        'Study_ID': study_id,
                        'Fold': fold,
                        'Trainer': trainer,
                        'Dice': calculate_dice(gt_data, pred_data),
                        'HD95': calculate_boundary_metrics(gt_data, pred_data),
                        'Sensitivity': calculate_sensitivity(gt_data, pred_data),
                        'GT_Volume_ml': gt_volume,
                        'Pred_Volume_ml': pred_volume,
                        'MAE(ml)': abs(pred_volume - gt_volume),
                        'RVD(ml)': (pred_volume - gt_volume) / gt_volume if gt_volume != 0 else 0,
                        'Precision': calculate_precision(gt_data, pred_data),
                        'AVD(ml)': abs(pred_volume - gt_volume),
                        'LDR': calculate_ldr(gt_data, pred_data),
                        'FPR': calculate_fpr(gt_data, pred_data),
                    })

                except Exception as e:
                    print(f"    error on {study_id} (fold {fold}): {e}")
                    continue

    if mismatches:
        print(f"  {len(mismatches)} shape mismatches skipped:")
        for study_id, fold, gt_shape, pred_shape in mismatches:
            print(f"    {study_id} fold {fold}: GT {gt_shape} vs pred {pred_shape}")

    return rows


def write_workbook(rows, output_path):
    df = pd.DataFrame(rows)
    with pd.ExcelWriter(output_path) as writer:
        df.to_excel(writer, sheet_name="All_Folds", index=False)

        if {'Fold', 'Trainer'}.issubset(df.columns):
            per_trainer_fold = df.groupby(['Trainer', 'Fold']).mean(numeric_only=True).reset_index()
            per_trainer_fold['Study_ID'] = 'Average_per_Trainer_Fold'

            per_trainer = df.groupby('Trainer').mean(numeric_only=True).reset_index()
            per_trainer['Fold'] = 'All_Folds'
            per_trainer['Study_ID'] = 'Average_per_Trainer'

            overall = df.mean(numeric_only=True)
            overall['Trainer'] = 'All_Trainers'
            overall['Fold'] = 'All_Folds'
            overall['Study_ID'] = 'Overall_Average'

            pd.concat([per_trainer_fold, per_trainer, pd.DataFrame([overall])]).to_excel(
                writer, sheet_name="Averages", index=False)
    return len(df)


def main():
    parser = argparse.ArgumentParser(
        description="Per-case segmentation metrics for every nnU-Net experiment and fold")
    parser.add_argument("--results_dir", required=True,
                        help="nnUNet_results: one subdirectory per experiment")
    parser.add_argument("--gt_base_dir", required=True,
                        help="nnUNet_preprocessed: <experiment>/gt_segmentations")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute experiments whose output file already exists")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for experiment in sorted(os.listdir(args.results_dir)):
        experiment_path = os.path.join(args.results_dir, experiment)
        if not os.path.isdir(experiment_path):
            continue

        output_path = os.path.join(args.output_dir, f"{experiment}_metrics.xlsx")
        if os.path.exists(output_path) and not args.overwrite:
            print(f"Skipping {experiment}: output exists. Use --overwrite to recompute.")
            continue

        gt_dir = os.path.join(args.gt_base_dir, experiment, "gt_segmentations")
        if not os.path.exists(gt_dir):
            print(f"GT directory not found for {experiment}: {gt_dir}")
            continue

        print(f"Processing {experiment}")
        rows = process_experiment(experiment_path, gt_dir, args.n_folds)

        if rows:
            n = write_workbook(rows, output_path)
            print(f"  wrote {n} rows to {output_path}")
        else:
            print(f"  no data for {experiment}")


if __name__ == "__main__":
    main()
