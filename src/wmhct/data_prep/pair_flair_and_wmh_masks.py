"""
Stage 1.5 - Pair each MSS3 FLAIR volume with its manually edited WMH mask.

Manuscript: Methods (reference label preparation)
Run: wmhct-pair-labels --help
"""

import argparse
import os
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Pair FLAIR volumes with WMH masks")
    parser.add_argument("--mri_dir", required=True)
    parser.add_argument("--wmh_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--mask_pattern", default="MSS3_ED_{pid}_WMH_edited.nii.gz",
                        help="Mask filename template; {pid} is replaced by <subject>_<visit>")
    args = parser.parse_args()

    plot_dir = os.path.join(args.out_dir, "plots")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    wmh_files = set(os.listdir(args.wmh_dir))
    paired, skipped = 0, 0

    for mri_file in sorted(os.listdir(args.mri_dir)):
        split_name = mri_file.split('_')
        if len(split_name) < 5:
            print(f"Unexpected filename format: {mri_file}")
            skipped += 1
            continue

        patient_id = split_name[2] + '_' + split_name[4]  # e.g. 028_V3
        wmh_file = args.mask_pattern.format(pid=patient_id)

        if wmh_file not in wmh_files:
            skipped += 1
            continue

        patient_folder = os.path.join(args.out_dir, f"MSS3_ED_{patient_id}")
        os.makedirs(patient_folder, exist_ok=True)

        shutil.copy(os.path.join(args.mri_dir, mri_file), patient_folder)
        shutil.copy(os.path.join(args.wmh_dir, wmh_file), patient_folder)

        wmh_img = nib.load(os.path.join(patient_folder, wmh_file)).get_fdata()
        n_positive = int((wmh_img > 0).sum())
        print(f"{patient_id}: mask values {np.unique(wmh_img)}, shape {wmh_img.shape}, "
              f"positive voxels {n_positive}")
        if n_positive == 0:
            print(f"  WARNING: {patient_id} has an empty mask.")

        plt.figure(figsize=(6, 6))
        plt.imshow(wmh_img[:, :, wmh_img.shape[2] // 2], cmap='binary')
        plt.title(f'WMH mask: {wmh_file}')
        plt.savefig(os.path.join(plot_dir, f"MSS3_ED_{patient_id}.png"))
        plt.close()
        paired += 1

    print(f"\nPaired {paired}, skipped {skipped}. QC montages in {plot_dir}")


if __name__ == "__main__":
    main()
