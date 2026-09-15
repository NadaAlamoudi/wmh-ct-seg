"""
Stage 1.4 - Apply the sequence classifier to a table of DICOM header values.

Manuscript: Methods (FLAIR identification).
Run: wmhct-classify-sequences --help
"""

import argparse

import pandas as pd

from wmhct.data_prep.classify_mri_sequence import MRIClassifier

REQUIRED_COLUMNS = ["Scanning Sequence", "Sequence Name", "TR", "TE", "FA", "TI"]


def main():
    parser = argparse.ArgumentParser(description="Classify MRI sequences from a DICOM header table")
    parser.add_argument("--in_csv", required=True, help="CSV of DICOM header values, one row per series")
    parser.add_argument("--out_csv", required=True, help="Where to write the annotated CSV")
    parser.add_argument("--encoding", default="latin1")
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv, encoding=args.encoding)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input table is missing required columns: {missing}")

    classifier = MRIClassifier()
    df["Modified MRI Sequence"] = df.apply(
        lambda row: classifier.classify(
            row["Scanning Sequence"], row["Sequence Name"],
            row["TR"], row["TE"], row["FA"], row["TI"]
        ),
        axis=1,
    )

    df.to_csv(args.out_csv, index=False)

    counts = df["Modified MRI Sequence"].value_counts(dropna=False)
    print(counts)
    unknown = counts.get("Unknown", 0) + counts.get("Unknown_EP", 0)
    print(f"\nUnresolved series: {unknown} of {len(df)} ({100 * unknown / len(df):.1f}%). "
          "These require manual inspection before FLAIR selection.")


if __name__ == "__main__":
    main()
