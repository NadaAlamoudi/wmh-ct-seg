"""
Stage 3.3 - Register CT volumes to a CT template (12-DOF affine).

Manuscript: Methods, template-space experiment
Run: wmhct-register-ct2tmpl --help
"""

import argparse
import os
import re
import subprocess
import time

from wmhct import config


def register_scans(template_name, ct_name, ct_reg_2_template, trans_matrix):
    flirt_command = [config.FLIRT,
                     '-in', ct_name,
                     '-ref', template_name,
                     '-out', ct_reg_2_template,
                     '-omat', trans_matrix,
                     '-dof', '12']  # affine

    print('Running command:', ' '.join(flirt_command))
    result = subprocess.run(flirt_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f'Error processing {ct_name}:')
        print(result.stderr.decode())
    else:
        print(f'File {ct_reg_2_template} processed successfully')


def main():
    parser = argparse.ArgumentParser(description='Register CT volumes to a CT template')
    parser.add_argument('--ct_dir', required=True)
    parser.add_argument('--template', default=str(config.CT_TEMPLATE))
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--id_pattern', default=r"(\d+)_")
    args = parser.parse_args()

    if not os.path.exists(args.template):
        raise FileNotFoundError(
            f"CT template not found: {args.template}. Set CT_TEMPLATE or pass --template.")

    os.makedirs(args.output_dir, exist_ok=True)
    patient_id_pattern = re.compile(args.id_pattern)

    ct_images = {}
    for file in os.listdir(args.ct_dir):
        if file.endswith(".nii.gz"):
            match = patient_id_pattern.search(file)
            if match and "CT" in file:
                ct_images[match.group(1)] = os.path.join(args.ct_dir, file)

    total_start_time = time.time()
    file_counter = 0

    for patient_id in sorted(ct_images):
        file_counter += 1
        ct_reg_2_template = os.path.join(args.output_dir, f"{patient_id}_CTreg2CT_template.nii.gz")
        trans_matrix = os.path.join(args.output_dir,
                                    f"{patient_id}_CTreg2CT_template_trans_matrix.mat")
        register_scans(args.template, ct_images[patient_id], ct_reg_2_template, trans_matrix)

    print(f"Registered {file_counter} volumes in {time.time() - total_start_time:.2f} seconds")


if __name__ == "__main__":
    main()
