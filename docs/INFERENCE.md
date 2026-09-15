# Running the model on a new non-contrast CT

## What it is

A 3D residual-encoder nnU-Net trained on NCCT with FLAIR-derived WMH labels transferred
into CT space. One input channel: the CT. No MRI at inference.

## 1. Install

```bash
pip install '.[nnunet]'
```

Also needs **SynthStrip via Docker** (`docs/INSTALL_EXTERNAL_TOOLS.md`). FSL and NiftyReg
are **not** needed for inference: it runs in native CT space and the output mask lands on
the input grid. A GPU is strongly recommended; on CPU add `-device cpu`.

## 2. Weights

Not publicly downloadable. Request access from **n.m.alamoudi@ed.ac.uk**, or the Row Fogo
Centre for Research Into Ageing and The Brain, University of Edinburgh
(<https://clinical-brain-sciences.ed.ac.uk/row-fogo-centre-for-research-into-ageing-and-the-brain>).
See the Data and model availability section of the README.

Once granted, the weights arrive as the nnU-Net results tree, which is what
`nnUNetv2_predict` expects:

```
Dataset101_WMH_CT/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/
├── dataset.json          <- required
├── plans.json            <- required
├── checksums.json
└── fold_{0..4}/checkpoint_final.pth
```

Check the transfer is complete:

```bash
wmhct-check-checkpoints --weights_dir .../nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres \
                        --manifest .../checksums.json
```

Then point nnU-Net at them:

```bash
export nnUNet_results=/absolute/path/to/weights
export nnUNet_raw=/tmp/nnunet_raw            # unused at inference, but must be set
export nnUNet_preprocessed=/tmp/nnunet_prep  # same
```

## 3. Preprocess

**Must match training exactly.** A mismatch degrades output silently — the model still
produces a plausible-looking mask. Order is **clip → skull strip → window**; windowing
zeroes everything outside 0–80 HU, so it goes last.

```bash
wmhct-convert-dicom --in_dir dicom/ --out_dir work/nifti          # if starting from DICOM
wmhct-clip-hu       --in_dir work/nifti       --out_dir work/ct_clipped
wmhct-skull-strip   -in work/ct_clipped       -out work/ct_stripped
wmhct-window-fixed  --in_dir work/ct_stripped --out_dir work/ct_windowed \
                    --window_center 40 --window_width 80
```

Use the **fixed** window. `wmhct-window-gmm` is the design-choice variant and is not what
these weights were trained with. There is no cropping and no registration.

nnU-Net needs a channel suffix — one channel means `_0000`:

```bash
mkdir -p work/nnunet_input
for f in work/ct_windowed/*.nii.gz; do
  cp "$f" "work/nnunet_input/$(basename "$f" .nii.gz)_0000.nii.gz"
done
```

## 4. Predict

```bash
nnUNetv2_predict -i work/nnunet_input -o work/predictions \
  -d 101 -c 3d_fullres -p nnUNetResEncUNetMPlans -tr nnUNetTrainer \
  -f 0 1 2 3 4 -chk checkpoint_final.pth
```

| Flag | Why |
|---|---|
| `-d 101` | Dataset101_WMH_CT, the fine-tuned model (MSS3 + CIM + IST-3) |
| `-c 3d_fullres` | the configuration the paper reports |
| `-p nnUNetResEncUNetMPlans` | residual-encoder plans; **omitting this loads the wrong plans** |
| `-tr nnUNetTrainer` | the default trainer, which is what was used |
| `-f 0 1 2 3 4` | five-fold ensemble, as evaluated in the paper |
| `-chk checkpoint_final.pth` | the checkpoint the reported results come from |

Also useful: `--save_probabilities`, `-device cpu`, `-npp 2 -nps 2` to reduce memory.
A single fold (`-f 0`) is ~5× faster at some cost in accuracy; say which you used.

## 5. Mask to burden

```bash
wmhct-wmh-volume --in_dir work/predictions --out_csv wmh_volumes.csv
```

Volume in mL per scan plus a burden class: low ≤ 10 mL, medium 10–25, high > 25.

Citation: see `CITATION.cff`. The reported results are the five-fold ensemble of
Dataset101_WMH_CT with `checkpoint_final.pth`.
