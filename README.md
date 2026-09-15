# wmh-ct-seg

Code for the paper "AI-Based Pipeline for the Segmentation of White Matter
Hypoattenuations in CT Scans: A Design-Choice Validation Study".

Model weights and participant data are not distributed here. See
[Data and model availability](#data-and-model-availability).

The manuscript is under revision at PLOS Digital Health. Citation details will be added
on acceptance; see `CITATION.cff`.

## Scope

This repository contains the full pipeline for segmenting white matter hypoattenuation
(WMH) on non-contrast CT: DICOM conversion, CT preprocessing, FLAIR-to-CT registration to
propagate MRI-derived reference labels, nnU-Net training, and the statistical analyses
reported in the paper. Registration is required only to build the training labels; at
inference the model takes a single non-contrast CT volume and requires no paired MRI.

The model is intended for WMH burden estimation and severity stratification. It is not a
replacement for MRI-based voxel-level segmentation: agreement with MRI-derived reference
volumes is strong at the level of total burden, but voxel-level overlap on CT is
intrinsically limited by the low attenuation contrast between WMH and normal-appearing
white matter.

It was developed and evaluated on three cohorts: MSS3 and IST-3, both comprising patients
with ischaemic stroke, and CIM, a cohort of healthy adults. Evaluation used internal
cross-validation on MSS3 only. Performance on other populations has not been tested.


## Installation

```bash
conda env create -f env.yml
conda activate wmhct
pip install .
```

Four external tools are called as subprocesses and are not installed by conda:

| Tool | Used for | Version |
|---|---|---|
| dcm2niix | DICOM to NIfTI conversion, gantry tilt correction | see release notes |
| FSL | FLIRT registration, `convert_xfm` transform inversion | 6.0 |
| NiftyReg | `reg_aladin`, `reg_resample`, `reg_transform` | 1.5.77 |
| FreeSurfer SynthStrip | skull stripping | via Docker |

Point `wmhct.config` at them with environment variables:

```bash
export WMH_DATA_ROOT=/path/to/data
export WMH_RESULTS_ROOT=/path/to/results
export FSL_DIR=/usr/local/fsl
export NIFTYREG_DIR=/usr/local/niftyreg
export DCM2NIIX=$(which dcm2niix)
```

A GPU is needed only for nnU-Net training, not for this pipeline.

## Usage

Installing the package puts one command on the PATH per pipeline stage. Run any with
`--help`. `example_pipeline.sh` gives the full order.

```
wmhct-convert-dicom        DICOM to NIfTI
wmhct-classify-sequences   identify FLAIR series from DICOM headers
wmhct-clip-hu              clip to the valid Hounsfield range
wmhct-window-fixed         brain window
wmhct-skull-strip          SynthStrip
wmhct-register-flair2ct    FLAIR to CT, propagate reference masks
wmhct-dataset-json         nnU-Net dataset definition
wmhct-inverse-to-flair     carry predictions back to FLAIR space
wmhct-volumes              volume table
wmhct-figures-10-11        volume agreement figures
```

- **Running the model on your own CT scans: `docs/INFERENCE.md`**
- External tools (dcm2niix, FSL, NiftyReg, Docker/SynthStrip): `docs/INSTALL_EXTERNAL_TOOLS.md`
- nnU-Net training: `docs/TRAINING.md`
- The statistical tests, with the command that produces each: `docs/statistics_index.md`

## Data and model availability

**The participant imaging data and the model weights are not distributed in this
repository.** MSS3 and IST-3 are governed by data-sharing agreements that do not permit
public release of individual-level data or of models derived from it.

Requests for data from either cohort, and for the trained model weights, should be directed
to the **Row Fogo Centre for Research into Ageing and the Brain**, University of Edinburgh.
Contact **n.m.alamoudi@ed.ac.uk**, or see
<https://clinical-brain-sciences.ed.ac.uk/row-fogo-centre-for-research-into-ageing-and-the-brain>.

Apply using the forms in `docs/`: **`S1_File_MSS3_data_request_form.doc`** and
**`S2_File_IST3_data_request_form.docx`**.

## Licence

GPL-3.0, except `src/wmhct/nnunet/generate_dataset_json.py`, which is derived from
nnU-Net and remains under Apache-2.0. See `THIRD_PARTY_NOTICES.md`.
