"""
Verify a downloaded nnU-Net weights folder before running inference.

Run: wmhct-check-checkpoints --help
"""

import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path

# Files nnU-Net needs alongside the fold folders in order to predict.
REQUIRED_TOP_LEVEL = ["dataset.json", "plans.json"]

EXPECTED_FOLDS = [0, 1, 2, 3, 4]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check_checkpoint(path):
    """Return (ok, size_bytes, message) for one .pth file."""
    if not path.exists():
        return False, 0, "missing"
    size = path.stat().st_size
    if size == 0:
        return False, 0, "empty file"
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            broken = zf.testzip()
        if broken is not None:
            return False, size, f"corrupt member: {broken}"
        if not names:
            return False, size, "archive has no entries"
        return True, size, f"complete, {len(names)} entries"
    except zipfile.BadZipFile:
        return False, size, ("truncated or corrupt download "
                             "(zip header present, no central directory)")
    except Exception as exc:  # pragma: no cover
        return False, size, f"{type(exc).__name__}: {exc}"


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights_dir", required=True, type=Path,
                    help="the nnUNetTrainer__<plans>__<config> folder holding fold_0..fold_4")
    ap.add_argument("--checkpoint", default="checkpoint_final.pth",
                    choices=["checkpoint_final.pth", "checkpoint_best.pth", "both"],
                    help="which checkpoint nnU-Net will load (default: checkpoint_final.pth)")
    ap.add_argument("--folds", nargs="*", type=int, default=EXPECTED_FOLDS,
                    help="folds to require (default: 0 1 2 3 4)")
    ap.add_argument("--sha256", action="store_true",
                    help="also print a SHA-256 for each complete file, for the manifest")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="compare against a checksums.json written by --write_manifest")
    ap.add_argument("--write_manifest", type=Path, default=None,
                    help="write checksums.json for the files that pass, to upload alongside "
                         "the weights")
    args = ap.parse_args()

    root = args.weights_dir
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return 1

    problems = []

    for name in REQUIRED_TOP_LEVEL:
        if not (root / name).is_file():
            problems.append(f"missing {name} in {root}")
            print(f"MISSING  {name}")
        else:
            print(f"OK       {name}")

    wanted = (["checkpoint_final.pth", "checkpoint_best.pth"]
              if args.checkpoint == "both" else [args.checkpoint])

    digests = {}
    print()
    for fold in args.folds:
        fold_dir = root / f"fold_{fold}"
        if not fold_dir.is_dir():
            problems.append(f"missing fold_{fold}")
            print(f"MISSING  fold_{fold}/")
            continue
        for ckpt in wanted:
            path = fold_dir / ckpt
            ok, size, message = check_checkpoint(path)
            label = "OK      " if ok else "FAIL    "
            print(f"{label} fold_{fold}/{ckpt:22s} {size / 1e6:8.1f} MB  {message}")
            if not ok:
                problems.append(f"fold_{fold}/{ckpt}: {message}")
            elif args.sha256 or args.write_manifest:
                digests[f"fold_{fold}/{ckpt}"] = {"sha256": sha256(path), "bytes": size}
                if args.sha256:
                    print(f"         sha256 {digests[f'fold_{fold}/{ckpt}']['sha256']}")

    if args.write_manifest and digests:
        args.write_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.write_manifest.write_text(json.dumps(digests, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {args.write_manifest} with {len(digests)} entries.")

    if args.manifest:
        expected = json.loads(args.manifest.read_text())
        print()
        for key, want in sorted(expected.items()):
            path = root / key
            if not path.is_file():
                problems.append(f"{key}: listed in the manifest but not present")
                print(f"MISSING  {key}")
                continue
            got = sha256(path)
            if got != want["sha256"]:
                problems.append(f"{key}: checksum mismatch")
                print(f"MISMATCH {key}\n         expected {want['sha256']}\n"
                      f"         got      {got}")
            else:
                print(f"OK       {key} checksum matches")

    print()
    if problems:
        print(f"{len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
        print("\nA truncated checkpoint is a broken download, not a broken model. "
              "Re-download that file; a resumable transfer such as "
              "`gdown --continue` or rclone is more reliable than a browser for "
              "files of this size.")
        return 1

    print("All requested checkpoints are present and complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
