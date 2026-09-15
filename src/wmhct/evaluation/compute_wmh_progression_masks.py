"""
Stage 6.5 - Voxelwise WMH change between visits: stable, growth and regression.

Manuscript: Methods 2.8.3. Produces WMHstable_vol, WMHgrow_vol and WMHshrink_vol, the
    six outcomes compared between the CT-MRI and MRI-MRI groups in Section 3.5 and
    regressed in Section 3.6
Run: wmhct-wmh-progression --help
"""

import argparse
import os

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import label

from wmhct import config

# 26-connectivity, as in the original.
STRUCTURE = np.ones((3, 3, 3), dtype=int)


def load_binary(path):
    """Load a mask and binarise it. Any non-zero voxel is foreground."""
    img = nib.load(path)
    return (img.get_fdata() > 0).astype(np.float32), img


def geometry_matches(img_a, img_b, data_a, data_b):
    return data_a.shape == data_b.shape and np.allclose(img_a.affine, img_b.affine)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--participants_xlsx", default=str(config.CT_AT_V0_LIST),
                    help="spreadsheet whose first column holds the participant IDs")
    ap.add_argument("--v0_mask_dir", default=str(config.MODEL7_PREDICTIONS_FLAIR_SPACE),
                    help="V0 WMH masks, named <digits>_nnUNet_in_FLAIR_space.nii.gz")
    ap.add_argument("--v1_mask_dir", default=str(config.WMH_MASKS_FLAIR_V1),
                    help="V1 reference WMH masks, named <digits>_gt.nii.gz")
    ap.add_argument("--out_mask_dir", default=str(config.WMH_PROGRESSION_MASKS),
                    help="where the three difference masks per participant are written")
    ap.add_argument("--out_xlsx", default=str(config.WMH_PROGRESSION_TABLE),
                    help="per-participant volumes and cluster counts")
    ap.add_argument("--no_write_masks", action="store_true",
                    help="compute the table only, do not save the difference masks")
    args = ap.parse_args()

    participants = (pd.read_excel(args.participants_xlsx)
                    .iloc[:, 0].dropna().astype(str).tolist())
    print(f"{len(participants)} participants listed in {args.participants_xlsx}")

    results = []
    skipped = {"missing_v0": 0, "missing_v1": 0, "shape": 0, "affine": 0}

    for participant in participants:
        # The mask filenames are keyed by the numeric part of the participant ID:
        # MSS3_ED_901 -> 901.
        digits = ''.join(filter(str.isdigit, participant.split('_')[2]))
        v0_path = os.path.join(args.v0_mask_dir, f'{digits}_nnUNet_in_FLAIR_space.nii.gz')
        v1_path = os.path.join(args.v1_mask_dir, f'{digits}_gt.nii.gz')

        if not os.path.exists(v0_path):
            print(f'{participant}: no V0 mask at {v0_path}. Skipping.')
            skipped["missing_v0"] += 1
            continue
        if not os.path.exists(v1_path):
            print(f'{participant}: no V1 mask at {v1_path}. Skipping.')
            skipped["missing_v1"] += 1
            continue

        v0, v0_img = load_binary(v0_path)
        v1, v1_img = load_binary(v1_path)

        if v0.shape != v1.shape:
            print(f'{participant}: mask shapes differ. Skipping.')
            skipped["shape"] += 1
            continue
        if not np.allclose(v0_img.affine, v1_img.affine):
            print(f'{participant}: mask affines differ. Skipping.')
            skipped["affine"] += 1
            continue

        # Normal-appearing white matter is the complement of the WMH mask.
        nawm_v0 = 1 - v0
        nawm_v1 = 1 - v1

        stable = v0 * v1           # WMH at both visits
        regress = v0 * nawm_v1     # WMH at V0 that is not WMH at V1
        growth = nawm_v0 * v1      # not WMH at V0, WMH at V1

        if not args.no_write_masks:
            out_dir = os.path.join(args.out_mask_dir, participant,
                                   'WMHprogression_V0_to_V1')
            os.makedirs(out_dir, exist_ok=True)
            for arr, name in ((stable, 'WMHremain'),
                              (regress, 'WMHregress'),
                              (growth, 'WMHgrowth')):
                nib.save(nib.Nifti1Image(arr, v0_img.affine, v0_img.header),
                         os.path.join(out_dir, f'{participant}_V1_{name}.nii.gz'))

        voxel_volume = float(np.prod(v0_img.header.get_zooms()[:3]))

        results.append({
            'PatientID': participant,
            'WMH_vol_v0': v0.sum() * voxel_volume,
            'WMH_vol_v1': v1.sum() * voxel_volume,
            'WMHremain_vol': stable.sum() * voxel_volume,
            'WMHregress_vol': regress.sum() * voxel_volume,
            'WMHgrowth_vol': growth.sum() * voxel_volume,
            'WMHremain_count': label(stable, structure=STRUCTURE)[1],
            'WMHregress_count': label(regress, structure=STRUCTURE)[1],
            'WMHgrowth_count': label(growth, structure=STRUCTURE)[1],
        })

    results_df = pd.DataFrame(results)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_xlsx)), exist_ok=True)
    results_df.to_excel(args.out_xlsx, index=False)

    print(f'\nWrote {len(results_df)} of {len(participants)} participants '
          f'to {args.out_xlsx}')
    if any(skipped.values()):
        print('Skipped: ' + ', '.join(f'{k}={v}' for k, v in skipped.items() if v))
        print('State the excluded participants and the reason in the Methods.')


if __name__ == "__main__":
    main()
