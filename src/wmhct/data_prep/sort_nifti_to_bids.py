"""
Stage 1.2 - Sort converted NIfTI files into BIDS format.

Manuscript: Methods, data organisation
Run: wmhct-sort-bids --help
"""

import argparse
import csv
import os
import shutil

from tqdm import tqdm


def parse_filename(filename):
    """Split <ID>_<study>_<date>_<modality>_... into its components."""
    parts = filename.split("_")
    if len(parts) >= 4:
        return {
            "Study_ID": parts[0],
            "Study_Description": parts[1],
            "scan_date": parts[2],
            "Modality": parts[3],
        }
    return None


def get_modality_folder(filename):
    return "CT" if "CT" in filename else None


def get_sub_folder(filename):
    return "anat" if "CT" in filename else None


def convert_nifti_to_bids(input_folder, output_folder):
    all_files = [(root, fn) for root, dirs, files in os.walk(input_folder) for fn in files]
    pbar = tqdm(total=len(all_files), desc="Processing files", ncols=70)

    for root, dirs, files in os.walk(input_folder):
        pbar.update(1)
        for filename in files:
            full_path = os.path.join(root, filename)
            info = parse_filename(filename)

            if info is None:
                print(f"Skipping {filename}: unexpected filename format.")
                continue

            modality_folder = get_modality_folder(filename)
            sub2_folder = get_sub_folder(filename)
            if modality_folder is None or sub2_folder is None:
                print(f"Skipping {filename}: unknown modality.")
                continue

            output_dir = os.path.join(output_folder, f"sub-{info['Study_ID']}",
                                      modality_folder, f"ses-{info['scan_date']}", sub2_folder)
            os.makedirs(output_dir, exist_ok=True)
            shutil.copy2(full_path, os.path.join(output_dir, filename))

    pbar.close()


def main():
    parser = argparse.ArgumentParser(description="Sort NIfTI files into a BIDS-like tree")
    parser.add_argument("--in_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--participants_csv", default="participants.csv")
    args = parser.parse_args()

    if not os.path.exists(args.participants_csv):
        with open(args.participants_csv, "w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=["subject_Num", "participant_id"]).writeheader()

    convert_nifti_to_bids(args.in_dir, args.out_dir)


if __name__ == "__main__":
    main()
