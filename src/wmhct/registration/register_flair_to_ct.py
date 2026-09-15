"""
Stage 3.1 - Register FLAIR to CT and propagate the WMH mask into CT space.

Manuscript: Methods, CT-MRI registration
Run: wmhct-register-flair2ct --help
"""

import argparse
import os
import re
import subprocess
import time

from wmhct import config


def apply_transformation_to_mask(mask_name, ct_name, transformed_mask_name, trans_matrix, tool):
    if tool == 'flirt':
        command = [config.FLIRT,
                   '-in', mask_name,
                   '-ref', ct_name,
                   '-out', transformed_mask_name,
                   '-init', trans_matrix,
                   '-applyxfm',
                   '-interp', 'nearestneighbour']
    elif tool == 'niftyreg':
        command = [config.REG_RESAMPLE,
                   '-ref', ct_name,
                   '-flo', mask_name,
                   '-res', transformed_mask_name,
                   '-trans', trans_matrix,
                   '-inter', '0']
    else:
        raise ValueError(f"unknown tool: {tool}")

    print('Running command:', ' '.join(command))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f'Error transforming mask {mask_name}:')
        print(result.stderr.decode())
        return False
    print(f'Mask {transformed_mask_name} transformed successfully')
    return True


def register_scans(flair_name, ct_name, flair_reg_2_ct, trans_matrix, tool):
    if tool == 'flirt':
        command = [config.FLIRT,
                   '-in', flair_name,
                   '-ref', ct_name,
                   '-out', flair_reg_2_ct,
                   '-omat', trans_matrix,
                   '-dof', '6',
                   '-cost', 'normmi']
    elif tool == 'niftyreg':
        command = [config.REG_ALADIN,
                   '-ref', ct_name,
                   '-flo', flair_name,
                   '-res', flair_reg_2_ct,
                   '-aff', trans_matrix]
    else:
        raise ValueError(f"unknown tool: {tool}")

    print('Running command:', ' '.join(command))
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f'Error registering to {ct_name}:')
        print(result.stderr.decode())
        return False
    print(f'File {flair_reg_2_ct} processed successfully')
    return True


def main():
    parser = argparse.ArgumentParser(description='Register FLAIR to CT and propagate WMH masks')
    parser.add_argument('--ct_dir', required=True, help='Directory containing the CT images')
    parser.add_argument('--mri_dir', required=True, help='Directory containing the FLAIR images')
    parser.add_argument('--mask_dir', required=True, help='Directory containing the WMH masks')
    parser.add_argument('--output_dir', required=True, help='Registered images and matrices')
    parser.add_argument('--trans_masks_dir', required=True, help='Transformed masks')
    parser.add_argument('--tool', required=True, choices=['flirt', 'niftyreg'])
    parser.add_argument('--id_pattern', default=r"(?:normalized_)?IST3_(\d+)_",
                        help='Regex with one capture group giving the participant ID')
    args = parser.parse_args()

    for d in (args.output_dir, args.trans_masks_dir):
        os.makedirs(d, exist_ok=True)

    patient_id_pattern = re.compile(args.id_pattern)
    ct_images, flair_images, masks = {}, {}, {}

    for file in os.listdir(args.ct_dir):
        if file.endswith(".nii.gz"):
            match = patient_id_pattern.search(file)
            if match and "CT" in file:
                ct_images.setdefault(match.group(1), []).append(os.path.join(args.ct_dir, file))

    for file in os.listdir(args.mri_dir):
        if file.endswith(".nii.gz"):
            match = patient_id_pattern.search(file)
            if match and "FLAIR" in file:
                flair_images[match.group(1)] = os.path.join(args.mri_dir, file)

    for file in os.listdir(args.mask_dir):
        if file.endswith(".nii.gz"):
            match = patient_id_pattern.search(file)
            if match and "FLAIR" in file:
                masks[match.group(1)] = os.path.join(args.mask_dir, file)

    print(f"Indexed {len(ct_images)} participants with CT, {len(flair_images)} with FLAIR, "
          f"{len(masks)} with masks")

    total_start_time = time.time()
    file_counter = 0

    for patient_id in sorted(flair_images):
        patient_start_time = time.time()

        if patient_id not in ct_images:
            print(f"No CT for participant {patient_id}; skipping.")
            continue
        if patient_id not in masks:
            print(f"No mask for participant {patient_id}; skipping.")
            continue

        for ct_file in ct_images[patient_id]:
            file_counter += 1
            ct_filename = os.path.basename(ct_file)
            stem = '_'.join(ct_filename.split('_')[:4])

            flair_reg_2_ct = os.path.join(args.output_dir, f"{stem}_FlairReg2CT.nii.gz")
            ext = '.mat' if args.tool == 'flirt' else '.txt'
            trans_matrix = os.path.join(args.output_dir, f"{stem}_FlairReg2CT_trans_matrix{ext}")
            transformed_mask_name = os.path.join(
                args.trans_masks_dir, f"{stem}_WMH_mask_in_CT_space.nii.gz")

            if register_scans(flair_images[patient_id], ct_file, flair_reg_2_ct,
                              trans_matrix, args.tool):
                apply_transformation_to_mask(masks[patient_id], ct_file, transformed_mask_name,
                                             trans_matrix, args.tool)

        print(f"Participant {patient_id}: {time.time() - patient_start_time:.2f} seconds")

    print(f"Registered {file_counter} CT sessions in "
          f"{time.time() - total_start_time:.2f} seconds")


if __name__ == "__main__":
    main()
