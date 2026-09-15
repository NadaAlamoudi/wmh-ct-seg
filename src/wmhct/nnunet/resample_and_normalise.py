"""
Stage 4.0 - Resampling and z-score intensity normalisation (preprocessing experiment).

Manuscript: Methods, preprocessing experiments
Run: wmhct-resample-normalise --help
"""

import argparse
import os

import nibabel as nib
import SimpleITK as sitk


def resample_to_common_shape(image, target_shape, interpolator=sitk.sitkLinear):
    """Resample a SimpleITK image onto a fixed grid of `target_shape` voxels."""
    new_spacing = [old_sz * old_spc / new_sz
                   for old_sz, old_spc, new_sz
                   in zip(image.GetSize(), image.GetSpacing(), target_shape)]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(target_shape)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def resample_image(image, out_spacing=(1.0, 1.0, 1.0), is_label=False):
    """Resample to a target voxel spacing; nearest neighbour for label maps."""
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(out_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())

    orig_size = image.GetSize()
    orig_spacing = image.GetSpacing()
    resampler.SetSize([int(round(orig_size[i] * (orig_spacing[i] / out_spacing[i])))
                       for i in range(len(orig_size))])
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkLinear)
    return resampler.Execute(image)


def z_score_normalization(img):
    data = img.get_fdata()
    return nib.Nifti1Image((data - data.mean()) / data.std(), img.affine)


def preprocess_image(input_path, output_path, target_shape):
    try:
        img_nib = nib.load(input_path)
        img_sitk = sitk.ReadImage(input_path)

        # Computed but not saved; see the caveat in the module docstring.
        resampled_image = resample_to_common_shape(img_sitk, target_shape)
        print("resampled size:", resampled_image.GetSize())

        nib.save(z_score_normalization(img_nib), output_path)
    except Exception as e:
        print(f"Failed to process {input_path}. Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Resample and z-score normalise registered CT/MRI volumes")
    parser.add_argument("--in_folder", type=str, required=True)
    parser.add_argument("--out_folder", type=str, required=True)
    parser.add_argument("--target_shape", type=int, nargs=3, default=[512, 512, 217])
    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    for counter, image_file in enumerate(sorted(os.listdir(args.in_folder)), start=1):
        if image_file.endswith('.nii.gz'):
            preprocess_image(os.path.join(args.in_folder, image_file),
                             os.path.join(args.out_folder, image_file),
                             tuple(args.target_shape))
            print(counter)


if __name__ == "__main__":
    main()
