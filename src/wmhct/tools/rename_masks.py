#!/usr/bin/env python3
"""
Rename the three mask sets to a single uniform convention.

Run: wmhct-rename-masks --help

Note: Dry run by default. --apply renames, --undo restores from the mapping it writes.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

# Default working root. These modules read folders of masks that sit beside each
# other, not inside the package, so the root defaults to the current directory and
# is overridable with --root.
ROOT = Path.cwd()
MAPPING_NAME = "rename_mapping.csv"

# folder -> (regex with the raw id as group 1, new suffix)
SETS = {
    "GT_WMHmasks_MSS3_in_FLAIR_space":
        (r"^MSS3_ED_(\d{3}(?:_\d)?)\.nii\.gz$", "ref"),
    "nnunet_Predicted_masks_in_FLAIR_space_invertred":
        (r"^(\d{3,4})_nnUNet_in_FLAIR_space\.nii\.gz$", "pred"),
    "Stroke_masks_in_FLAIR_space":
        (r"^(\d{3,4})_stroke_mask\.nii\.gz$", "stroke"),
    "FLAIR_MSS3":
        (r"^(\d{3,4})_FLAIR\.nii\.gz$", "flair"),
}


def canonical_id(raw: str) -> str:
    """'904_1' -> '904_1'; '9041' -> '904_1'; '904' -> '904'."""
    if "_" in raw:
        return raw
    if len(raw) == 3:
        return raw
    if len(raw) == 4:
        return f"{raw[:3]}_{raw[3]}"
    raise ValueError(f"unexpected identifier: {raw!r}")


def plan(folder: Path, pattern: str, suffix: str) -> list[tuple[Path, Path]]:
    # Accept either the original name or the already-renamed name, so that
    # re-running the script is idempotent rather than reporting zero matches.
    rx = re.compile(f"(?:{pattern})|(?:^(\\d{{3}}(?:_\\d)?)_{suffix}\\.nii\\.gz$)")
    pairs: list[tuple[Path, Path]] = []
    seen: dict[str, Path] = {}
    unmatched = []
    for f in sorted(folder.glob("*.nii.gz")):
        m = rx.match(f.name)
        if not m:
            unmatched.append(f.name)
            continue
        raw = m.group(1) if m.group(1) is not None else m.group(2)
        sid = canonical_id(raw)
        if sid in seen:
            raise SystemExit(f"ERROR: duplicate id {sid} in {folder.name}: "
                             f"{seen[sid].name} and {f.name}")
        seen[sid] = f
        pairs.append((f, folder / f"{sid}_{suffix}.nii.gz"))
    if unmatched:
        print(f"  note: {len(unmatched)} file(s) in {folder.name} did not match "
              f"the pattern and are left untouched: {unmatched[:5]}")
    return pairs


def apply_rename(pairs: list[tuple[Path, Path]], folder: Path) -> None:
    """Two-pass rename via temporary names, then write the mapping."""
    already = [(s, d) for s, d in pairs if s == d]
    todo = [(s, d) for s, d in pairs if s != d]
    if not todo:
        print(f"  {folder.name}: already in the target convention")
        return

    existing = {p.name for p in folder.glob("*.nii.gz")}
    targets = {d.name for _, d in todo}
    clash = targets & (existing - {s.name for s, _ in todo})
    if clash:
        raise SystemExit(f"ERROR: target name(s) already exist in {folder.name} "
                         f"and are not part of the rename: {sorted(clash)}")

    tmp: list[tuple[Path, Path, Path]] = []
    for i, (src, dst) in enumerate(todo):
        t = folder / f".__rename_tmp_{i}__.nii.gz"
        src.rename(t)
        tmp.append((src, t, dst))
    for src, t, dst in tmp:
        t.rename(dst)

    rows = [(s.name, d.name) for s, d in pairs]
    with (folder / MAPPING_NAME).open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["original_name", "new_name"])
        w.writerows(rows)
    print(f"  {folder.name}: renamed {len(todo)} file(s), "
          f"{len(already)} already correct, mapping -> {MAPPING_NAME}")


def undo(folder: Path) -> None:
    mp = folder / MAPPING_NAME
    if not mp.exists():
        print(f"  {folder.name}: no {MAPPING_NAME}, nothing to undo")
        return
    with mp.open() as fh:
        rows = list(csv.DictReader(fh))
    tmp = []
    for i, r in enumerate(rows):
        cur = folder / r["new_name"]
        if not cur.exists():
            print(f"  {folder.name}: missing {r['new_name']}, skipped")
            continue
        t = folder / f".__undo_tmp_{i}__.nii.gz"
        cur.rename(t)
        tmp.append((t, folder / r["original_name"]))
    for t, dst in tmp:
        t.rename(dst)
    mp.unlink()
    print(f"  {folder.name}: restored {len(tmp)} file(s)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--apply", action="store_true", help="perform the rename")
    ap.add_argument("--undo", action="store_true", help="revert using the mapping")
    args = ap.parse_args()

    if args.undo:
        for name in SETS:
            undo(args.root / name)
        return 0

    all_ids: dict[str, set[str]] = {}
    for name, (pattern, suffix) in SETS.items():
        folder = args.root / name
        if not folder.is_dir():
            raise SystemExit(f"ERROR: missing folder {folder}")
        pairs = plan(folder, pattern, suffix)
        all_ids[suffix] = {d.name.rsplit("_", 1)[0] for _, d in pairs}
        print(f"{name}: {len(pairs)} file(s)")
        for src, dst in pairs[:3]:
            print(f"    {src.name}  ->  {dst.name}")
        if len(pairs) > 3:
            print(f"    ... and {len(pairs) - 3} more")
        if args.apply:
            apply_rename(pairs, folder)
        print()

    common = set.intersection(*all_ids.values())
    for suffix, ids in all_ids.items():
        extra = ids - common
        if extra:
            print(f"WARNING: {len(extra)} id(s) present only in the "
                  f"'{suffix}' set: {sorted(extra)}", file=sys.stderr)
    print(f"{len(common)} subject id(s) present in all three sets")

    if not args.apply:
        print("\nDry run. Re-run with --apply to rename.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
