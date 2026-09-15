# External tools

Four programs pip cannot install. **None are needed for the statistical modules** — if you
only want to reproduce the tables and figures from the result tables, `pip install .` is
enough. Inference needs only Docker/SynthStrip; FSL and NiftyReg are for building training
data.

| Tool | Used for |
|---|---|
| dcm2niix | DICOM → NIfTI |
| FSL | FLIRT registration, `convert_xfm` inversion |
| NiftyReg | `reg_aladin` rigid+affine, `reg_resample`, `reg_transform` |
| SynthStrip (Docker) | skull stripping |

## dcm2niix

<https://github.com/rordenlab/dcm2niix> — `brew install dcm2niix`,
`apt-get install dcm2niix`, or `conda install -c conda-forge dcm2niix`.

## FSL 6.0

<https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index>

```bash
curl -fLO https://fsl.fmrib.ox.ac.uk/fsldownloads/fslconda/releases/fslinstaller.py
python3 fslinstaller.py
```

The installer edits your shell profile; the result should be equivalent to setting
`FSLDIR`, sourcing `$FSLDIR/etc/fslconf/fsl.sh` and adding `$FSLDIR/bin` to `PATH`. On
Apple Silicon some builds need Rosetta 2.

## NiftyReg 1.5.77

<https://github.com/KCL-BMEIS/niftyreg>. Easiest is `conda install -c conda-forge niftyreg`.
To build the exact version:

```bash
git clone https://github.com/KCL-BMEIS/niftyreg.git && cd niftyreg
git checkout v1.5.77
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/niftyreg -DCMAKE_BUILD_TYPE=Release -DUSE_OPENMP=ON ..
make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" && sudo make install
```

Needs a C++ compiler, CMake ≥ 3.2 and zlib. Registration used **default `reg_aladin`
parameters**: symmetric block matching, normalised cross-correlation over 4-voxel blocks,
rigid initialisation then a 12-DOF affine.

## Docker and SynthStrip

Docker Desktop (<https://www.docker.com/products/docker-desktop/>) or, on Linux,
`curl -fsSL https://get.docker.com | sudo sh`. Then:

```bash
docker pull freesurfer/synthstrip
docker images --digests freesurfer/synthstrip     # record the digest and pin it
```

<https://surfer.nmr.mgh.harvard.edu/docs/synthstrip/>. `wmhct-skull-strip` invokes the
container and mounts the directories for you.

## Environment

```bash
export WMH_DATA_ROOT=/path/to/imaging
export WMH_RESULTS_ROOT=/path/to/results
export FSL_DIR=/usr/local/fsl
source "$FSL_DIR/etc/fslconf/fsl.sh"
export NIFTYREG_DIR=/usr/local/niftyreg
export DCM2NIIX=/usr/local/bin/dcm2niix
export CT_TEMPLATE=/path/to/ct_template.nii.gz
export PATH="$FSL_DIR/bin:$NIFTYREG_DIR/bin:$PATH"
```

## CT template

Experiments 2 and 3 register CT to a standard template — the age-specific T1 and CT
templates of Rorden et al., <https://www.nitrc.org/projects/clinicaltbx/>. Set
`CT_TEMPLATE`. Not needed for the native-space experiments, including the final model.
