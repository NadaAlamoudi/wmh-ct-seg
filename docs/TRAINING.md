# nnU-Net training and inference

The segmentation model is nnU-Net used as distributed, without architectural
modification.

## Model sequence reported in the manuscript

| Model | Training data | Purpose |
|---|---|---|
| 1–5 | MSS3 CT, varying preprocessing and registration | design-choice comparison |
| 6 | Model 5 fine-tuned on CIM pseudo-labels | pseudo-label contribution |
| 7 | Model 5 fine-tuned on CIM and IST-3 pseudo-labels | final reported model |

Model 6 has the highest DSC and precision; Model 7 has the lowest MAE and the highest
sensitivity. The manuscript should not describe Model 7 as uniformly best.

## Commands

Read from the trained results tree of the final model:

| Setting | Value |
|---|---|
| nnU-Net | 2.3.1 |
| Dataset | `Dataset101_WMH_CT` (ID 101) |
| Plans | `nnUNetResEncUNetMPlans` |
| Configuration | `3d_fullres` |
| Trainer | `nnUNetTrainer` (default) |
| Folds | 5 |
| Training cases | 282 |

```bash
export nnUNet_raw="$WMH_RESULTS_ROOT/nnUNet_raw"
export nnUNet_preprocessed="$WMH_RESULTS_ROOT/nnUNet_preprocessed"
export nnUNet_results="$WMH_RESULTS_ROOT/nnUNet_results"

# 1. Plan and preprocess
nnUNetv2_plan_and_preprocess -d 101 --verify_dataset_integrity

# 2. Train five folds
for FOLD in 0 1 2 3 4; do
  nnUNetv2_train 101 3d_fullres $FOLD -tr nnUNetTrainer -p nnUNetResEncUNetMPlans
done

# 3. Predict
nnUNetv2_predict -i <IMAGES_TS> -o <PREDICTIONS> \
  -d 101 -c 3d_fullres -tr nnUNetTrainer -p nnUNetResEncUNetMPlans -f 0 1 2 3 4
```

Fine-tuning for Models 6 and 7 was performed by initialising from the previous model's
checkpoint (`-pretrained_weights <checkpoint>`). Record the
exact checkpoint paths, the learning rate and the number of fine-tuning epochs.

## Model weights

Weights are not stored in this repository. The model was
trained on MSS3, IST-3 and CIM imaging, and the data sharing agreements governing MSS3 and
IST-3 do not permit unrestricted release of the data or of models derived from it.

To request access, contact **n.m.alamoudi@ed.ac.uk**, or apply to the Row Fogo Centre for
Research Into Ageing and The Brain, University of Edinburgh:
<https://clinical-brain-sciences.ed.ac.uk/row-fogo-centre-for-research-into-ageing-and-the-brain>

**To run the model on your own scans, see `docs/INFERENCE.md`.**
