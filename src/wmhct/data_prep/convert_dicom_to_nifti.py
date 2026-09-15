"""
Stage 1.1 - DICOM to NIfTI conversion (for MSS3 CT).

Manuscript: Methods, data preparation
Run: wmhct-convert-dicom --help
"""

import argparse
import os
import subprocess
import time

import pydicom
from tqdm import tqdm

from wmhct import config


def convert_dicom_series_to_nifti_one_folder(input_folder, ct_folder, patient_name, cmd_path):
    output_filename = None
    try:
        dicom_files = [pydicom.dcmread(os.path.join(input_folder, filename))
                       for filename in os.listdir(input_folder) if filename.endswith('.dcm')]
        dicom_file = dicom_files[0]

        modality = dicom_file.Modality
        series_number = dicom_file.SeriesNumber
        if hasattr(dicom_file, 'SeriesDate'):
            scan_date = dicom_file.SeriesDate
        else:
            scan_date = dicom_file.StudyDate

        output_filename = f"{patient_name}_BRIC2_{scan_date}_{modality}_{series_number}.nii.gz"
        output_filename_prefix = output_filename[:output_filename.index('.nii.gz')]

        subprocess.run([cmd_path, "-f", output_filename_prefix, "-p", "y", "-z", "y",
                        "-o", ct_folder, input_folder], check=True)
        print(f"Conversion from DICOM to NIfTI completed: {output_filename}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting file: {output_filename}")
        print("Error message:", e.output)
    except Exception as e:
        error_msg = f"Error in generating NIfTI file: ({output_filename}) - Reason: {str(e)}"
        print(error_msg)
        with open('Error_log_for_conversion_DICOMtoNIFTI.txt', 'a') as log_file:
            print(error_msg, file=log_file)


def apply_function_to_subfolders(folder_path, cmd_path, min_slices=5):
    ct_folders_count = sum(1 for subdir, dirs, _ in os.walk(folder_path) if 'CT' in dirs)

    with tqdm(total=ct_folders_count, desc="Processing DICOM files") as pbar:
        for subdir, dirs, files in os.walk(folder_path):
            if 'CT' not in dirs:
                continue
            # Patient name comes from the folder name rather than the DICOM PatientName
            # tag, because some patients have more than one scan.
            patient_name = os.path.basename(subdir)
            ct_folder_path = os.path.join(subdir, 'CT')
            for ct_subdir, _, _ in os.walk(ct_folder_path):
                num_slices = sum(1 for f in os.listdir(ct_subdir) if f.endswith('.dcm'))
                if num_slices >= min_slices:
                    convert_dicom_series_to_nifti_one_folder(
                        ct_subdir, ct_folder_path, patient_name, cmd_path)
                    pbar.update(1)
                else:
                    print(f"Skipping {ct_subdir}: only {num_slices} slices.")


def main():
    parser = argparse.ArgumentParser(description="Convert CT DICOM series to NIfTI")
    parser.add_argument("--in_dir", default=str(config.MSS3_DICOM),
                        help="Root of the DICOM tree; each patient folder contains a CT/ folder")
    parser.add_argument("--dcm2niix", default=config.DCM2NIIX)
    parser.add_argument("--min_slices", type=int, default=5,
                        help="Series with fewer slices than this are skipped")
    args = parser.parse_args()

    start_time = time.time()
    apply_function_to_subfolders(args.in_dir, args.dcm2niix, args.min_slices)
    print(f"Total processing time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
