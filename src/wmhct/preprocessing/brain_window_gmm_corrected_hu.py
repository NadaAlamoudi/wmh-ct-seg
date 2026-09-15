"""
Stage 2.3b - Brain window with per-scan HU centre correction via a Gaussian mixture.

Run: wmhct-window-gmm --help

Note: The published model uses the fixed window (wmhct-window-fixed), not this variant.
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.mixture import GaussianMixture  # noqa: E402


def fit_brain_centre(data, nominal_centre=40.0, lo=-100.0, hi=300.0,
                     n_components=3, random_state=0):
    """Return (mean, sd) of the GMM component closest to the nominal window centre."""
    flat = data.flatten()
    filtered = flat[(flat > lo) & (flat < hi)].reshape(-1, 1)

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(filtered)

    idx = int(np.argmin(np.abs(gmm.means_ - nominal_centre)))
    brain_mean = float(gmm.means_[idx][0])
    brain_std = float(np.sqrt(gmm.covariances_[idx][0][0]))
    return brain_mean, brain_std


def apply_windowing(data, brain_mean, window_width):
    brain_hu_min = brain_mean - (window_width / 2)
    brain_hu_max = brain_mean + (window_width / 2)
    brain_mask = (data >= brain_hu_min) & (data <= brain_hu_max)
    return np.where(brain_mask, data, 0)


def main():
    parser = argparse.ArgumentParser(description="GMM-corrected brain windowing of CT volumes")
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--window_width", type=float, default=80.0)
    parser.add_argument("--nominal_centre", type=float, default=40.0)
    parser.add_argument("--qc_dir", default=None)
    parser.add_argument("--centres_csv", default="fitted_window_centres.csv")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.qc_dir:
        os.makedirs(args.qc_dir, exist_ok=True)

    rows = []
    count = 0

    for ct_filename in sorted(os.listdir(args.in_dir)):
        if not ct_filename.endswith(".nii.gz"):
            continue

        img = nib.load(os.path.join(args.in_dir, ct_filename))
        original_data = img.get_fdata().copy()

        brain_mean, brain_std = fit_brain_centre(original_data, nominal_centre=args.nominal_centre)
        data = apply_windowing(original_data, brain_mean, args.window_width)

        rows.append({"file": ct_filename,
                     "fitted_centre_hu": round(brain_mean, 3),
                     "fitted_sd_hu": round(brain_std, 3),
                     "offset_from_nominal_hu": round(brain_mean - args.nominal_centre, 3)})

        new_img = nib.Nifti1Image(data, img.affine, img.header)
        nib.save(new_img, os.path.join(
            args.out_dir, ct_filename.replace(".nii.gz", "_brain_window_correctedHU.nii.gz")))

        if args.qc_dir:
            slice_idx = original_data.shape[2] // 2
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            ax[0].imshow(original_data[:, :, slice_idx], cmap=plt.cm.bone)
            ax[0].set_title(f"Original ({ct_filename})")
            ax[1].imshow(data[:, :, slice_idx], cmap=plt.cm.bone)
            ax[1].set_title(f"Windowed, centre = {brain_mean:.1f} HU")
            fig.savefig(os.path.join(args.qc_dir, ct_filename.replace(".nii.gz", ".png")), dpi=120)
            plt.close(fig)

        count += 1

    if not rows:
        print("No .nii.gz files found in", args.in_dir)
        return

    with open(args.centres_csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    offsets = np.array([r["offset_from_nominal_hu"] for r in rows])
    print(f"Total number of processed CT files: {count}")
    print(f"Fitted centre offset from {args.nominal_centre:.0f} HU: "
          f"mean {offsets.mean():+.2f}, SD {offsets.std(ddof=1):.2f}, "
          f"range [{offsets.min():+.2f}, {offsets.max():+.2f}] HU")
    print(f"Fitted window centres written to {args.centres_csv}")


if __name__ == "__main__":
    main()
