# Released results

This folder holds the per-scan five-fold cross-validation metrics for the final model (Model 7: trained on MSS3,
fine-tuned with CIM and IST-3), evaluated on the 91 MSS3 CT scans; see the "Data and model availability" section of the top-level README.

## `S1_crossvalidation_results.csv`

| Column | |
|---|---|
| `participant` | non-identifying code, `Patient_001` … `Patient_080` |
| `session` | 1 for a participant's only scan, otherwise 1, 2, 3 in order |
| `fold` | cross-validation fold, 0–4 |
| `predicted_volume_ml` | WMH volume from the CT-derived segmentation, **in CT space** |
| `dice` | Dice similarity coefficient against the FLAIR-derived reference |
| `hd95_mm` | 95th-percentile Hausdorff distance |
| `sensitivity` | recall against the reference |
| `precision` | positive predictive value |
| `absolute_volume_error_ml` | absolute difference between predicted and reference volume |
| `absolute_volume_difference_ml` | as above |
| `relative_volume_difference` | signed, as a fraction of the reference |

All volumes are measured **in native CT space**, which is where the model predicts and
where these metrics were computed.

Requests for the imaging data, the per-scan tables behind the paper analyses, and the
trained model weights go to the **Row Fogo Centre for Research Into Ageing and The Brain**,
University of Edinburgh. See the top-level README.
