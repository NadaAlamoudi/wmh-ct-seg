# Third-party code and licences

This repository is released under GPL-3.0 (see `LICENSE`), with the following exception.

## nnU-Net (DKFZ) — Apache Licence 2.0

`src/wmhct/nnunet/generate_dataset_json.py` contains `generate_dataset_json` and its
helper `get_identifiers_from_splitted_files`, taken from the nnU-Net project developed by
the Division of Medical Image Computing, German Cancer Research Center (DKFZ),
Heidelberg. That code is licensed under the Apache Licence, Version 2.0. The Apache
header at the top of the file must be retained, and the file remains subject to
Apache-2.0.

Apache-2.0 is one-way compatible with GPL-3.0: Apache-2.0 code may be incorporated into a
GPL-3.0 work, and the combined work is distributed under GPL-3.0. The reverse is not
true. Keeping the header is what makes this compliant.

Source: https://github.com/MIC-DKFZ/nnUNet

The segmentation model is nnU-Net, used as distributed. No architectural changes were
made. Cite:

> Isensee F, Jaeger PF, Kohl SAA, Petersen J, Maier-Hein KH. nnU-Net: a self-configuring
> method for deep learning-based biomedical image segmentation. *Nature Methods*
> 2021;18:203–211.

## External tools invoked as subprocesses

No source from these projects is included in this repository. They must be installed
separately and are subject to their own licences.

**FreeSurfer SynthStrip** — skull stripping, via the `freesurfer/synthstrip` Docker
image, under the FreeSurfer licence. Cite:

> Hoopes A, Mora JS, Dalca AV, Fischl B, Hoffmann M. SynthStrip: skull-stripping for any
> brain image. *NeuroImage* 2022;260:119474.

**FSL** — `flirt` and `convert_xfm`. Licensed for non-commercial use; commercial users
require a licence from Oxford University Innovation. Cite:

> Jenkinson M, Bannister P, Brady M, Smith S. Improved optimization for the robust and
> accurate linear registration and motion correction of brain images. *NeuroImage*
> 2002;17:825–841.

**NiftyReg** — `reg_aladin`, `reg_resample`, `reg_transform`. BSD-style licence. Cite:

> Modat M, Cash DM, Daga P, Winston GP, Duncan JS, Ourselin S. Global image registration
> using a symmetric block-matching approach. *Journal of Medical Imaging* 2014;1:024003.

**dcm2niix** — DICOM conversion. BSD-2-Clause. Cite:

> Li X, Morgan PS, Ashburner J, Smith J, Rorden C. The first step for neuroimaging data
> analysis: DICOM to NIfTI conversion. *Journal of Neuroscience Methods* 2016;264:47–56.
