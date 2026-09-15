"""
Build the one table that may be released, from the per-case metrics workbook.

Run: wmhct-build-release-table --help

"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

# Deny by default. A column reaches the release table only by being named here.
ALLOWED_COLUMNS = {
    "Fold": "fold",
    "Dice": "dice",
    "HD95": "hd95_mm",
    "Sensitivity": "sensitivity",
    "Precision": "precision",
    "Pred_Volume_ml": "predicted_volume_ml",
    "MAE(ml)": "absolute_volume_error_ml",
    "RVD(ml)": "relative_volume_difference",
    "AVD(ml)": "absolute_volume_difference_ml",
}

# Columns that, combined with predicted volume, recover the reference volume.
RECONSTRUCTABLE = {"Sensitivity", "Precision", "MAE(ml)", "RVD(ml)", "AVD(ml)"}

# Never released under any option. Listed explicitly so the refusal is visible in the code
# rather than implied by omission.
FORBIDDEN_ALWAYS = {
    "GT_Volume_ml",          # the reference segmentation volume
    "WMHvol", "oldSLvol_r", "indexSLvo", "oldSLvol_mm3", "indexSLvol_mm3",
    "newSLvolV1_mm3",        # stroke lesion volumes
    "oldStrokeType", "indexStrokeType",
    "Age", "Sex", "Time_Difference_Days", "V0_Time", "V1_Time",
    "V1_PVLOverall", "V1_DWMLOverall",   # Fazekas subscales
}

# Three-digit participant, optional one-digit session, with or without an underscore and
# with an optional cohort prefix. The prefix must be matched and discarded rather than
# stripped by joining digits: "MSS3" itself contains a digit, so joining every digit in
# "MSS3_903" yields "3176" and silently reassigns the scan to participant 317.
CASE_RE = re.compile(r"^(?:MSS3_)?(?:ED_)?(?P<pid>\d{3})(?:_?(?P<session>\d))?$")


def participant_and_session(study_id: str):
    """
    'MSS3_9021' -> ('902', 2) ; 'MSS3_903' -> ('903', 1) ; 'MSS3_ED_901_1' -> ('901', 2).

    Same rule as tools/rename_masks.py. Raises rather than
    guessing, because a silently mis-parsed identifier merges two participants into one
    release code.
    """
    m = CASE_RE.match(str(study_id).strip())
    if not m:
        raise ValueError(
            f"Cannot parse participant and session from {study_id!r}. Expected a "
            f"three-digit study number with an optional one-digit session, such as "
            f"MSS3_903, MSS3_9021 or MSS3_ED_901_1.")
    session = m.group("session")
    return m.group("pid"), (int(session) + 1 if session is not None else 1)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics_xlsx", required=True,
                    help="per-case metrics workbook for the released model")
    ap.add_argument("--sheet", default="All_Folds")
    ap.add_argument("--id_col", default="Study_ID")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--mapping_csv", default=None,
                    help="where to write the study-ID to release-code key. Store it with "
                         "the study data, NOT with the released table")
    ap.add_argument("--prefix", default="Patient")
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--suppress_reconstructable", action="store_true",
                    help="also drop sensitivity, precision and the error terms, so the "
                         "reference volume cannot be recovered arithmetically")
    ap.add_argument("--decimals", type=int, default=4)
    args = ap.parse_args()

    df = pd.read_excel(args.metrics_xlsx, sheet_name=args.sheet)
    if args.id_col not in df.columns:
        raise SystemExit(f"No column {args.id_col!r}. Columns: {list(df.columns)}")

    allowed = dict(ALLOWED_COLUMNS)
    if args.suppress_reconstructable:
        for col in RECONSTRUCTABLE:
            allowed.pop(col, None)

    # --- identifiers ---------------------------------------------------
    parsed = df[args.id_col].map(participant_and_session)
    df["_participant"] = [p for p, _ in parsed]
    df["_session"] = [s for _, s in parsed]

    order = sorted(df["_participant"].unique())
    codes = {p: f"{args.prefix}_{i:0{args.width}d}" for i, p in enumerate(order, start=1)}
    df["participant"] = df["_participant"].map(codes)

    # --- columns -------------------------------------------------------
    present = [c for c in df.columns if c in allowed]
    dropped = [c for c in df.columns
               if c not in allowed and c not in {args.id_col, "_participant",
                                                 "_session", "participant"}]
    forbidden_seen = sorted(set(dropped) & FORBIDDEN_ALWAYS)
    other_dropped = sorted(set(dropped) - FORBIDDEN_ALWAYS)

    out = pd.DataFrame({"participant": df["participant"], "session": df["_session"]})
    n_sessions = df.groupby("_participant")["_session"].transform("size")
    out["session"] = df["_session"].where(n_sessions > 1, 1)
    for col in present:
        out[allowed[col]] = df[col]

    numeric = out.select_dtypes("number").columns
    out[numeric] = out[numeric].round(args.decimals)
    out = out.sort_values(["participant", "session"]).reset_index(drop=True)

    # --- report --------------------------------------------------------
    print(f"Source: {args.metrics_xlsx} [{args.sheet}]")
    print(f"{len(out)} scans from {out['participant'].nunique()} participants, "
          f"{int((n_sessions > 1).sum())} scans from repeat-scan participants\n")
    print("Released columns:")
    for col in present:
        note = "  (permits reference-volume recovery)" if col in RECONSTRUCTABLE else ""
        print(f"  {allowed[col]:32s} <- {col}{note}")
    if forbidden_seen:
        print("\nDropped, never releasable:")
        for col in forbidden_seen:
            print(f"  {col}")
    if other_dropped:
        print("\nDropped, not on the allow-list:")
        for col in other_dropped:
            print(f"  {col}")

    if not args.suppress_reconstructable and (RECONSTRUCTABLE & set(present)):
        print("\nNOTE: this table permits arithmetic recovery of the reference volume "
              "per scan.\n      That residual was raised and accepted; see "
              "results/README.md.\n      "
              "--suppress_reconstructable implements the stricter reading.")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\nWritten: {args.out_csv}")

    if args.mapping_csv:
        mapping = (df[["_participant", "participant", args.id_col, "_session"]]
                   .rename(columns={"_participant": "study_participant",
                                    args.id_col: "study_id",
                                    "_session": "session"})
                   .sort_values(["participant", "session"]))
        Path(args.mapping_csv).parent.mkdir(parents=True, exist_ok=True)
        mapping.to_csv(args.mapping_csv, index=False)
        print(f"Mapping:  {args.mapping_csv}")
        print("          This is the re-identification key. Store it with the study data "
              "under the same\n          access control. Never publish it alongside the "
              "released table.")
    else:
        print("\nNo --mapping_csv given, so no re-identification key was written. "
              "Pass one if you\nneed to trace a release code back to a study identifier.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
