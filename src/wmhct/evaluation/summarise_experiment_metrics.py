"""
Stage 6.0b - Add means and standard deviations to the per-case metric workbooks.

Manuscript: Table V (mean +/- SD per experiment)
Run: wmhct-summarise-experiments --help
"""

import argparse
import os
import shutil

import pandas as pd


def summarise(all_folds_df):
    """Mean and SD by trainer/fold, by trainer, and overall."""
    numeric_columns = all_folds_df.select_dtypes(include=[float, int]).columns

    per_trainer_fold = all_folds_df.groupby(['Trainer', 'Fold'])[numeric_columns].agg(
        ['mean', 'std']).reset_index()
    per_trainer_fold.columns = ['_'.join(c).strip() if c[1] else c[0]
                                for c in per_trainer_fold.columns.values]
    per_trainer_fold['Study_ID'] = 'Average_per_Trainer_Fold'

    per_trainer = all_folds_df.groupby('Trainer')[numeric_columns].agg(
        ['mean', 'std']).reset_index()
    per_trainer.columns = ['_'.join(c).strip() if c[1] else c[0]
                           for c in per_trainer.columns.values]
    per_trainer['Fold'] = 'All_Folds'
    per_trainer['Study_ID'] = 'Average_per_Trainer'

    overall = all_folds_df[numeric_columns].agg(['mean', 'std']).T
    overall.columns = ['mean', 'std']
    overall = overall.unstack().to_frame().T
    overall.columns = [f"{c[0]}_{c[1]}" for c in overall.columns]
    overall['Trainer'] = 'All_Trainers'
    overall['Fold'] = 'All_Folds'
    overall['Study_ID'] = 'Overall_Average'

    all_columns = (set(per_trainer_fold.columns) | set(per_trainer.columns)
                   | set(overall.columns))
    frames = [df[[c for c in all_columns if c in df.columns]]
              for df in (per_trainer_fold, per_trainer, overall)]
    combined = pd.concat(frames, ignore_index=True)

    ordered = ['Trainer', 'Fold', 'Study_ID']
    for metric in numeric_columns:
        ordered += [f"{metric}_mean", f"{metric}_std"]
    return combined.reindex(columns=ordered)


def main():
    parser = argparse.ArgumentParser(
        description="Add mean and SD sheets to the per-case metric workbooks")
    parser.add_argument("--in_dir", required=True,
                        help="Directory of <experiment>_metrics.xlsx files")
    parser.add_argument("--out_dir", default=None,
                        help="Where to write the annotated copies. Default: alongside "
                             "the inputs with a _with_std suffix")
    parser.add_argument("--in_place", action="store_true",
                        help="Write the new sheet back into the input file, as the "
                             "original script did")
    args = parser.parse_args()

    out_dir = args.out_dir or args.in_dir
    os.makedirs(out_dir, exist_ok=True)

    processed = 0
    for filename in sorted(os.listdir(args.in_dir)):
        if not filename.endswith(".xlsx"):
            continue

        in_path = os.path.join(args.in_dir, filename)
        try:
            all_folds_df = pd.read_excel(in_path, sheet_name="All_Folds")
        except ValueError:
            print(f"Skipping {filename}: no All_Folds sheet")
            continue

        summary = summarise(all_folds_df)

        if args.in_place:
            target = in_path
        else:
            target = os.path.join(out_dir, filename.replace(".xlsx", "_with_std.xlsx"))
            if not os.path.exists(target):
                shutil.copy2(in_path, target)

        with pd.ExcelWriter(target, mode='a', if_sheet_exists='replace') as writer:
            summary.to_excel(writer, sheet_name="Averages_with_Std", index=False)

        print(f"Averages_with_Std written to {target}")
        processed += 1

    print(f"\n{processed} workbook(s) processed.")
    if processed:
        print("Reminder: the SD in Table V is case-level, not a standard error. See the "
              "module docstring before describing differences between experiments.")


if __name__ == "__main__":
    main()
