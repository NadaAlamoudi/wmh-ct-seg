"""
Stage 5.1 - Project predictions back into native FLAIR space.

Manuscript: Methods, evaluation in native FLAIR space; Figures 10-11
Run: wmhct-inverse-to-flair --help
"""

import argparse
import csv
import os
import subprocess

import nibabel as nib
import numpy as np

from wmhct import config


def dice_score(mask1, mask2):
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    sum_masks = np.sum(mask1 > 0) + np.sum(mask2 > 0)
    return 1.0 if sum_masks == 0 else (2 * intersection) / sum_masks


def iou_score(mask1, mask2):
    intersection = np.sum((mask1 > 0) & (mask2 > 0))
    union = np.sum((mask1 > 0) | (mask2 > 0))
    return 1.0 if union == 0 else intersection / union


def calculate_volume(mask, voxel_volume):
    """Lesion volume in mL, from the voxel count and the voxel volume in mm3."""
    return (np.sum(mask > 0) * voxel_volume) / 1000


def invert_transformation_matrix(trans_matrix, inverted_matrix):
    result = subprocess.run([config.CONVERT_XFM, '-omat', inverted_matrix,
                             '-inverse', trans_matrix],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error inverting {trans_matrix}:", result.stderr.decode())
        return False
    return True


def invert_transformation_matrix_niftyreg(trans_matrix, inverted_matrix, ref_image):
    result = subprocess.run([config.REG_TRANSFORM, '-ref', ref_image, '-invAff',
                             trans_matrix, inverted_matrix],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error inverting {trans_matrix} with NiftyReg:", result.stderr.decode())
        return False
    return True


def apply_inverse_transformation(tool, nnunet_mask_ct, flair_image,
                                 transformed_mask_flair, inverted_matrix):
    if tool == 'flirt':
        command = [config.FLIRT, '-in', nnunet_mask_ct, '-ref', flair_image,
                   '-out', transformed_mask_flair, '-init', inverted_matrix,
                   '-applyxfm', '-interp', 'nearestneighbour']
    else:
        command = [config.REG_RESAMPLE, '-ref', flair_image, '-flo', nnunet_mask_ct,
                   '-res', transformed_mask_flair, '-trans', inverted_matrix, '-inter', '0']

    print('Running command:', ' '.join(command))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        print(f"Error transforming {nnunet_mask_ct}:", result.stderr.decode())
        return False
    return True


def process_all_files(tool, nnunet_ct_dir, flair_dir, flair_gt_dir,
                      trans_matrix_dir, output_dir, csv_file):
    os.makedirs(output_dir, exist_ok=True)
    ext = '.mat' if tool == 'flirt' else '.txt'
    n_done, n_skipped = 0, 0

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Patient ID', 'Dice Score', 'IoU Score',
                         'nnUNet Volume (ml)', 'GT Volume (ml)'])

        for nnunet_file in sorted(f for f in os.listdir(nnunet_ct_dir) if f.endswith(".nii.gz")):
            patient_id = nnunet_file.split("_")[0]

            nnunet_mask_ct = os.path.join(nnunet_ct_dir, nnunet_file)
            flair_image = os.path.join(flair_dir, f"{patient_id}_FLAIR.nii.gz")
            flair_gt_mask = os.path.join(flair_gt_dir, f"{patient_id}_gt.nii.gz")
            trans_matrix = os.path.join(trans_matrix_dir,
                                        f"{patient_id}_FlairReg2CT_trans_matrix{ext}")
            inverted_matrix = os.path.join(
                trans_matrix_dir, f"{patient_id}_FlairReg2CT_trans_matrix_inverted{ext}")
            transformed_mask_flair = os.path.join(
                output_dir, f"{patient_id}_nnUNet_in_FLAIR_space.nii.gz")

            if not all(os.path.exists(f) for f in
                       [nnunet_mask_ct, flair_image, flair_gt_mask, trans_matrix]):
                print(f"Skipping {patient_id}: missing files.")
                n_skipped += 1
                continue

            ok = (invert_transformation_matrix(trans_matrix, inverted_matrix) if tool == 'flirt'
                  else invert_transformation_matrix_niftyreg(trans_matrix, inverted_matrix,
                                                             flair_image))
            if not ok or not apply_inverse_transformation(
                    tool, nnunet_mask_ct, flair_image, transformed_mask_flair, inverted_matrix):
                n_skipped += 1
                continue

            nnunet_mask_flair = nib.load(transformed_mask_flair).get_fdata()
            flair_gt = nib.load(flair_gt_mask).get_fdata()
            voxel_volume = float(np.prod(nib.load(flair_image).header.get_zooms()[:3]))

            dice = dice_score(nnunet_mask_flair, flair_gt)
            iou = iou_score(nnunet_mask_flair, flair_gt)
            nnunet_volume = calculate_volume(nnunet_mask_flair, voxel_volume)
            gt_volume = calculate_volume(flair_gt, voxel_volume)

            writer.writerow([patient_id, f"{dice:.4f}", f"{iou:.4f}",
                             f"{nnunet_volume:.4f}", f"{gt_volume:.4f}"])
            print(f"{patient_id}: Dice={dice:.4f}, IoU={iou:.4f}, "
                  f"pred={nnunet_volume:.4f} mL, GT={gt_volume:.4f} mL")
            n_done += 1

    print(f"\nProcessed {n_done}, skipped {n_skipped}. Results in {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Move CT-space predictions into native FLAIR space and score them")
    parser.add_argument("--nnunet_ct_dir", required=True)
    parser.add_argument("--flair_dir", required=True)
    parser.add_argument("--flair_gt_dir", required=True)
    parser.add_argument("--trans_matrix_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--csv_file", default=None)
    parser.add_argument("--tool", choices=["flirt", "niftyreg"], required=True)
    args = parser.parse_args()

    csv_file = args.csv_file or f"nnunet_results_{args.tool}.csv"
    process_all_files(args.tool, args.nnunet_ct_dir, args.flair_dir, args.flair_gt_dir,
                      args.trans_matrix_dir, args.output_dir, csv_file)


if __name__ == "__main__":
    main()
