#!/usr/bin/env python3
"""
Section 3.7 - stroke-lesion contamination of predicted WMH.

Manuscript: Section 3.7, lesion overlap; Figure 14; the contamination figures quoted in
    the Results (median 0.08%, IQR 0.00-1.96%, 10 of 82 scans above 5%)

Run: wmhct-stroke-overlap --help

"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib

N_BOOTSTRAP = 10000
RNG_SEED = 20260814

PRED_DIR = "nnunet_Predicted_masks_in_FLAIR_space_invertred"
REF_DIR = "GT_WMHmasks_MSS3_in_FLAIR_space"
STROKE_DIR = "Stroke_masks_in_FLAIR_space"


# ----------------------------------------------------------------------------
# input
# ----------------------------------------------------------------------------

def index(folder: Path, suffix: str) -> dict[str, Path]:
    """Map subject id -> file for files named <ID>_<suffix>.nii.gz."""
    tail = f"_{suffix}.nii.gz"
    return {f.name[: -len(tail)]: f for f in sorted(folder.glob(f"*{tail}"))}


def load_binary(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    return np.asarray(img.dataobj) > 0, img


def check_geometry(a: nib.Nifti1Image, b: nib.Nifti1Image, name: str,
                   sid: str, tol: float = 1e-3) -> None:
    """Index-wise set operations are only meaningful on a shared grid."""
    if a.shape[:3] != b.shape[:3]:
        raise ValueError(f"[{sid}] shape mismatch for {name}: "
                         f"{b.shape[:3]} vs {a.shape[:3]}")
    if not np.allclose(a.affine, b.affine, atol=tol):
        raise ValueError(f"[{sid}] affine mismatch for {name}")


# ----------------------------------------------------------------------------
# per-subject
# ----------------------------------------------------------------------------

def dice(a: np.ndarray, b: np.ndarray) -> float:
    n = int(a.sum()) + int(b.sum())
    return 2.0 * float((a & b).sum()) / n if n else np.nan


def analyse(sid: str, p_path: Path, g_path: Path, s_path: Path) -> dict:
    P, p_img = load_binary(p_path)
    G, g_img = load_binary(g_path)
    S, s_img = load_binary(s_path)
    check_geometry(p_img, g_img, "reference", sid)
    check_geometry(p_img, s_img, "stroke", sid)

    vox_ml = float(np.prod(p_img.header.get_zooms()[:3])) / 1000.0
    n_p, n_g, n_s = int(P.sum()), int(G.sum()), int(S.sum())

    n_p_in_s = int((P & S).sum())
    n_g_in_s = int((G & S).sum())
    n_unmatched = int((P & S & ~G).sum())

    P_ex, G_ex = P & ~S, G & ~S
    n_p_ex, n_g_ex = int(P_ex.sum()), int(G_ex.sum())

    return {
        "subject": sid,
        "vol_pred_ml": n_p * vox_ml,
        "vol_ref_ml": n_g * vox_ml,
        "vol_stroke_ml": n_s * vox_ml,
        "n_pred_vox": n_p,
        "n_ref_vox": n_g,
        "n_stroke_vox": n_s,
        "stroke_present": n_s > 0,

        "n_pred_in_stroke": n_p_in_s,
        "n_ref_in_stroke": n_g_in_s,
        "n_pred_in_stroke_unmatched": n_unmatched,
        "vol_pred_in_stroke_ml": n_p_in_s * vox_ml,
        "f_pred_in_stroke": n_p_in_s / n_p if n_p else np.nan,
        "f_ref_in_stroke": n_g_in_s / n_g if n_g else np.nan,
        "f_unmatched_in_stroke": n_unmatched / n_p if n_p else np.nan,

        "dice_all": dice(P, G),
        "dice_excl_lesion": dice(P_ex, G_ex),
        "vol_diff_ml_all": (n_p - n_g) * vox_ml,
        "vol_diff_ml_excl_lesion": (n_p_ex - n_g_ex) * vox_ml,
    }


# ----------------------------------------------------------------------------
# cohort summary
# ----------------------------------------------------------------------------

def boot_ci_ratio(num: np.ndarray, den: np.ndarray) -> tuple[float, float]:
    """Percentile bootstrap CI over subjects for the pooled ratio sum/sum."""
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, num.size, size=(N_BOOTSTRAP, num.size))
    draws = num[idx].sum(axis=1) / den[idx].sum(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def boot_ci_median(v: np.ndarray) -> tuple[float, float]:
    v = v[np.isfinite(v)]
    if v.size < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.integers(0, v.size, size=(N_BOOTSTRAP, v.size))
    return tuple(np.percentile(np.median(v[idx], axis=1), [2.5, 97.5]))


def pct(x: float) -> str:
    return "n/a" if not np.isfinite(x) else f"{100 * x:.2f}%"


def iqr(s: pd.Series, f: str = ".2f") -> str:
    return f"{s.median():{f}} ({s.quantile(.25):{f}}-{s.quantile(.75):{f}})"


def summarise(df: pd.DataFrame) -> str:
    L: list[str] = []
    a = L.append
    pos = df[df.stroke_present]

    a("Stroke-lesion contamination of predicted WMH")
    a("=" * 62)
    a(f"Subjects analysed            : {len(df)}")
    a(f"Non-empty stroke mask        : {len(pos)}")
    a(f"Empty stroke mask            : {len(df) - len(pos)}")
    a("")
    a(f"Predicted WMH volume, median (IQR) : {iqr(df.vol_pred_ml)} mL")
    a(f"Reference WMH volume, median (IQR) : {iqr(df.vol_ref_ml)} mL")
    a(f"Stroke lesion volume, median (IQR) : {iqr(df.vol_stroke_ml)} mL")
    a("")

    n_g_in_s = int(df.n_ref_in_stroke.sum())
    a(f"Reference WMH voxels inside the lesion, whole cohort: {n_g_in_s}")
    if n_g_in_s == 0:
        a("  Reference WMH and stroke are mutually exclusive labels, so every")
        a("  predicted voxel inside the lesion is a false positive and")
        a("  f_unmatched_in_stroke is identical to f_pred_in_stroke.")
    a("")

    a("-" * 62)
    a("Contamination")
    a("-" * 62)
    pooled = df.n_pred_in_stroke.sum() / df.n_pred_vox.sum()
    lo, hi = boot_ci_ratio(df.n_pred_in_stroke.to_numpy(float),
                           df.n_pred_vox.to_numpy(float))
    a(f"Pooled fraction of predicted WMH inside the lesion : "
      f"{pct(pooled)}  95% CI [{pct(lo)}, {pct(hi)}]")
    a(f"Total predicted volume inside lesions              : "
      f"{df.vol_pred_in_stroke_ml.sum():.2f} mL of "
      f"{df.vol_pred_ml.sum():.2f} mL predicted")
    a("")

    f = pos.f_pred_in_stroke
    mlo, mhi = boot_ci_median(f.to_numpy(float))
    a(f"Per-subject fraction, lesion-positive subjects (n={len(pos)}):")
    a(f"  median {pct(f.median())} [IQR {pct(f.quantile(.25))}-"
      f"{pct(f.quantile(.75))}], 95% CI of median [{pct(mlo)}, {pct(mhi)}]")
    a(f"  maximum {pct(f.max())}")
    for t in (0.01, 0.05, 0.10):
        a(f"  subjects above {t:.0%}: {int((f > t).sum())}/{len(pos)}")
    a("")
    a("Largest contributors (share of all contaminating voxels):")
    top = df.nlargest(5, "n_pred_in_stroke")
    tot = df.n_pred_in_stroke.sum()
    for _, r in top.iterrows():
        a(f"  {r.subject:>6}  {r.vol_pred_in_stroke_ml:6.2f} mL  "
          f"{pct(r.f_pred_in_stroke):>7} of its predicted volume  "
          f"({100 * r.n_pred_in_stroke / tot:4.1f}% of cohort total, "
          f"lesion {r.vol_stroke_ml:.1f} mL)")
    a("")

    a("-" * 62)
    a("Agreement with and without the lesion territory")
    a("-" * 62)
    a(f"Dice, all voxels        median {iqr(df.dice_all, '.3f')}")
    a(f"Dice, lesion excluded   median {iqr(df.dice_excl_lesion, '.3f')}")
    a(f"Volume difference (pred - ref), all      "
      f"median {df.vol_diff_ml_all.median():+.3f} mL, "
      f"mean {df.vol_diff_ml_all.mean():+.3f} mL")
    a(f"Volume difference (pred - ref), excluded "
      f"median {df.vol_diff_ml_excl_lesion.median():+.3f} mL, "
      f"mean {df.vol_diff_ml_excl_lesion.mean():+.3f} mL")
    a("")
    return "\n".join(L)


# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path.cwd(),
                    help="folder holding the three mask sets "
                         "(default: current directory)")
    ap.add_argument("--out-prefix", default="wmh_stroke_overlap")
    args = ap.parse_args()

    pred = index(args.root / PRED_DIR, "pred")
    ref = index(args.root / REF_DIR, "ref")
    stroke = index(args.root / STROKE_DIR, "stroke")

    ids = sorted(set(pred) & set(ref) & set(stroke))
    if not ids:
        raise SystemExit("no subjects matched; run rename_masks.py --apply first")
    missing = (set(pred) | set(ref) | set(stroke)) - set(ids)
    print(f"matched {len(ids)} subjects "
          f"(pred {len(pred)}, ref {len(ref)}, stroke {len(stroke)})")
    if missing:
        print(f"WARNING: not present in all three sets: {sorted(missing)}")

    rows = []
    for i, sid in enumerate(ids, 1):
        rows.append(analyse(sid, pred[sid], ref[sid], stroke[sid]))
        if i % 20 == 0 or i == len(ids):
            print(f"  {i}/{len(ids)}", flush=True)

    df = pd.DataFrame(rows).sort_values("subject").reset_index(drop=True)
    csv_path = args.root / f"{args.out_prefix}_per_subject.csv"
    txt_path = args.root / f"{args.out_prefix}_summary.txt"
    df.to_csv(csv_path, index=False)

    text = summarise(df)
    txt_path.write_text(text + "\n")
    print()
    print(text)
    print(f"written: {csv_path}")
    print(f"written: {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
