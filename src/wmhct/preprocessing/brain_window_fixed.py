"""
Stage 2.3 - Fixed brain window (WL = 40 HU, WW = 80 HU), applied after skull stripping.

Manuscript: Methods, CT preprocessing
Run: wmhct-window-fixed --help
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402


def apply_windowing(data, window_center, window_width):
    """Zero everything outside the window; keep original HU inside it."""
    min_val = window_center - (window_width / 2)
    max_val = window_center + (window_width / 2)
    brain_mask = (data >= min_val) & (data <= max_val)
    return np.where(brain_mask, data, 0)


def main():
    parser = argparse.ArgumentParser(description="Apply a fixed brain window to CT volumes")
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--window_center", type=float, default=40.0)
    parser.add_argument("--window_width", type=float, default=80.0)
    parser.add_argument("--qc_dir", default=None, help="Optional directory for before/after PNGs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.qc_dir:
        os.makedirs(args.qc_dir, exist_ok=True)

    count = 0
    for ct_filename in sorted(os.listdir(args.in_dir)):
        if not ct_filename.endswith(".nii.gz"):
            continue

        img = nib.load(os.path.join(args.in_dir, ct_filename))
        original_data = img.get_fdata().copy()
        data = apply_windowing(original_data, args.window_center, args.window_width)

        new_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(new_img, os.path.join(
            args.out_dir, ct_filename.replace(".nii.gz", "_brain_window.nii.gz")))

        if args.qc_dir:
            slice_idx = original_data.shape[2] // 2
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(original_data[:, :, slice_idx], cmap='gray')
            ax[0].set_title(f"Original ({ct_filename})")
            ax[1].imshow(data[:, :, slice_idx], cmap='gray')
            ax[1].set_title(f"Windowed (WW={args.window_width:.0f}, WL={args.window_center:.0f})")
            fig.savefig(os.path.join(args.qc_dir, ct_filename.replace(".nii.gz", ".png")), dpi=120)
            plt.close(fig)

        count += 1

    print(f"Total number of files processed: {count}")


if __name__ == "__main__":
    main()
