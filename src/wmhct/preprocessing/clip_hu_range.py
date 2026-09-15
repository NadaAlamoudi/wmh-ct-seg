"""
Stage 2.1 - Clip CT to [-1024, 3071] HU and reset the header scaling.

Manuscript: Methods, CT preprocessing
Run: wmhct-clip-hu --help
"""

import argparse
import os

import nibabel as nib
import numpy as np


def clip_ct(ct_numpy, min_hu=-1024, max_hu=3071):
    return np.clip(ct_numpy, min_hu, max_hu)


def main():
    parser = argparse.ArgumentParser(description="Clip CT volumes to a valid HU range")
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--min_hu", type=float, default=-1024)
    parser.add_argument("--max_hu", type=float, default=3071)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    count = 0

    for ct_filename in sorted(os.listdir(args.in_dir)):
        if not ct_filename.endswith(".nii.gz"):
            continue

        img = nib.load(os.path.join(args.in_dir, ct_filename))
        data = clip_ct(img.get_fdata(), args.min_hu, args.max_hu)

        new_img = nib.Nifti1Image(data, img.affine, img.header)
        new_img.header['scl_slope'] = 1.0
        new_img.header['scl_inter'] = 0.0

        new_filename = ct_filename.replace(".nii.gz", "_HU_corrected.nii.gz")
        nib.save(new_img, os.path.join(args.out_dir, new_filename))
        count += 1

    print(f"Total number of files processed: {count}")


if __name__ == "__main__":
    main()
