"""
Stage 3.2 - Apply an existing FLAIR-to-CT transform to a WMH mask.

Manuscript: Methods, CT-MRI registration
Run: wmhct-register-masks2ct --help
"""

import argparse
import os
import re
import subprocess
import time

from wmhct import config


def apply_transformation_to_mask(mask_name, ct_name, transformed_mask_name, trans_matrix):
    flirt_command = [config.FLIRT,
                     '-in', mask_name,
                     '-ref', ct_name,
                     '-out', transformed_mask_name,
                     '-init', trans_matrix,
                     '-applyxfm',
                     '-interp', 'nearestneighbour']

    print('Running command:', ' '.join(flirt_command))
    result = subprocess.run(flirt_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f'Error transforming mask {mask_name}:')
        print(result.stderr.decode())
    else:
        print(f'Mask {transformed_mask_name} transformed successfully')


def main():
    parser = argparse.ArgumentParser(description='Resample WMH masks into CT space')
    parser.add_argument('--ct_dir', required=True)
    parser.add_argument('--mask_dir', required=True)
    parser.add_argument('--trans_masks_output_dir', required=True)
    parser.add_argument('--trans_matrix_dir', required=True)
    parser.add_argument('--id_pattern', default=r"(\d+)_")
    args = parser.parse_args()

    os.makedirs(args.trans_masks_output_dir, exist_ok=True)
    patient_id_pattern = re.compile(args.id_pattern)

    ct_images, masks = {}, {}

    for file in os.listdir(args.ct_dir):
        if file.endswith(".nii.gz"):
            match = patient_id_pattern.search(file)
            if match and "CT" in file:
                ct_images[match.group(1)] = os.path.join(args.ct_dir, file)

    for file in os.listdir(args.mask_dir):
        if file.endswith(".nii.gz"):
            match = patient_id_pattern.search(file)
            if match and "gt" in file:
                masks[match.group(1)] = os.path.join(args.mask_dir, file)

    total_start_time = time.time()
    file_counter = 0

    for patient_id in sorted(masks):
        if patient_id not in ct_images:
            print(f"No CT for participant {patient_id}; skipping.")
            continue

        file_counter += 1
        trans_matrix_path = os.path.join(
            args.trans_matrix_dir, f"{patient_id}_FlairReg2CT_trans_matrix.mat")
        transformed_mask_name = os.path.join(
            args.trans_masks_output_dir, f"{patient_id}_WMHmask_in_CT_space.nii.gz")
        apply_transformation_to_mask(masks[patient_id], ct_images[patient_id],
                                     transformed_mask_name, trans_matrix_path)

    print(f"Transformed {file_counter} masks in {time.time() - total_start_time:.2f} seconds")


if __name__ == "__main__":
    main()
