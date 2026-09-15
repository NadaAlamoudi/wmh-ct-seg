"""
Stage 4.2 - Write a custom nnU-Net splits_final.json.

Manuscript: not used for the reported MSS3 cross-validation; see the note below
Run: wmhct-make-splits --help

Note: The deposited splits restrict validation to the expert-labelled MSS3 scans.
"""

import argparse
import json
import os

import numpy as np


def create_splits(train_data_folder, fold_validation_cases, out_path, num_folds=5, seed=0):
    """
    Write an nnU-Net splits_final.json with explicit validation cases per fold.

    Parameters
    ----------
    train_data_folder : str
        Folder containing the training images, used to enumerate all case IDs.
    fold_validation_cases : dict[int, list[str]]
        Fold index -> case identifiers (without extension) held out in that fold.
    out_path : str
        Where to write splits_final.json.
    seed : int
        Seed for the shuffle of the training pool, so the file is reproducible.
    """
    rng = np.random.default_rng(seed)
    all_cases = [f.split('.')[0] for f in os.listdir(train_data_folder) if f.endswith('.nii.gz')]

    splits = []
    for i in range(num_folds):
        if i not in fold_validation_cases:
            raise ValueError(f"No validation cases specified for fold {i}")

        validation_cases = fold_validation_cases[i]
        missing = [c for c in validation_cases if c not in all_cases]
        if missing:
            raise ValueError(f"Validation cases not present in the dataset: {missing}")

        training_cases_pool = [c for c in all_cases if c not in validation_cases]
        rng.shuffle(training_cases_pool)

        splits.append({'train': training_cases_pool, 'val': validation_cases})

    with open(out_path, 'w') as fh:
        json.dump(splits, fh, indent=4)

    print(f"Generated {num_folds} folds with custom validation cases in '{out_path}'.")


def stratified_group_folds(case_ids, participant_ids, burden_categories,
                           num_folds=5, seed=0):
    """
    Participant-grouped, burden-stratified fold assignment.

    This is the design recommended in the limitations, NOT the one used for the reported
    results. No participant appears in both the training and validation split of a fold.

    Returns
    -------
    dict[int, list[str]] mapping fold index to validation case IDs, suitable for
    passing to `create_splits`.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    case_ids = np.asarray(case_ids)
    splitter = StratifiedGroupKFold(n_splits=num_folds, shuffle=True, random_state=seed)

    folds = {}
    for fold, (_, val_idx) in enumerate(
            splitter.split(case_ids, y=burden_categories, groups=participant_ids)):
        folds[fold] = case_ids[val_idx].tolist()
    return folds


def main():
    parser = argparse.ArgumentParser(description="Write a custom nnU-Net splits_final.json")
    parser.add_argument("--train_data_folder", required=True)
    parser.add_argument("--fold_spec_json", required=True,
                        help='JSON mapping fold index (as a string) to a list of case IDs')
    parser.add_argument("--out_path", default="splits_final.json")
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("NOTE: the reported cross-validation used nnU-Net's default random folds. "
          "Using a custom splits file produces a different partitioning.")

    with open(args.fold_spec_json) as fh:
        spec = {int(k): v for k, v in json.load(fh).items()}

    create_splits(args.train_data_folder, spec, args.out_path, args.num_folds, args.seed)


if __name__ == "__main__":
    main()
