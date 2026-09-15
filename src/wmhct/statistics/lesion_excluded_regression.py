#!/usr/bin/env python3
"""
Section 3.7 - does the lesion-volume effect on DSC survive removing the lesion?

Manuscript: Section 3.7, the change in the standardised coefficient for combined stroke
    lesion volume from -0.44 to -0.36 when the lesion territory is removed from both the
    predicted and the reference mask
Run: wmhct-lesion-excluded --help
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Default working root. These modules read folders of masks that sit beside each
# other, not inside the package, so the root defaults to the current directory and
# is overridable with --root.
ROOT = Path.cwd()


def fit(df: pd.DataFrame, outcome: str) -> dict:
    """OLS of DSC on stroke lesion volume and reference WMH volume, in mm^3."""
    X = sm.add_constant(df[["stroke_mm3", "wmh_mm3"]])
    m = sm.OLS(df[outcome], X).fit()
    sy = df[outcome].std()
    out = {"outcome": outcome, "n": int(m.nobs), "r2": m.rsquared}
    for v in ("stroke_mm3", "wmh_mm3"):
        ci = m.conf_int().loc[v]
        out[v] = {
            "b_per_10ml": m.params[v] * 1e4,
            "ci_per_10ml": (ci[0] * 1e4, ci[1] * 1e4),
            "beta": m.params[v] * df[v].std() / sy,
            "p": m.pvalues[v],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path,
                    default=ROOT / "wmh_stroke_overlap_per_subject.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df["stroke_mm3"] = df.vol_stroke_ml * 1000.0
    df["wmh_mm3"] = df.vol_ref_ml * 1000.0

    L: list[str] = []
    a = L.append
    a("Association between stroke lesion volume and DSC, before and after")
    a("removing the lesion territory from both masks")
    a("=" * 72)
    a(f"n = {len(df)} scans; predictors: combined stroke lesion volume, reference WMH volume")
    a("")

    for outcome, label in (("dice_all", "DSC, full masks"),
                           ("dice_excl_lesion", "DSC, lesion territory removed")):
        r = fit(df, outcome)
        a(f"{label}   (R2 = {r['r2']:.3f})")
        for v, name in (("stroke_mm3", "stroke lesion volume"),
                        ("wmh_mm3", "reference WMH volume")):
            s = r[v]
            a(f"  {name:22s} b = {s['b_per_10ml']:+.4f} per 10 mL"
              f"  95% CI [{s['ci_per_10ml'][0]:+.4f}, {s['ci_per_10ml'][1]:+.4f}]"
              f"  beta = {s['beta']:+.3f}  p = {s['p']:.3g}")
        a("")

    b_all = fit(df, "dice_all")["stroke_mm3"]["beta"]
    b_exc = fit(df, "dice_excl_lesion")["stroke_mm3"]["beta"]
    a(f"Standardised coefficient for stroke lesion volume: "
      f"{b_all:+.3f} -> {b_exc:+.3f} "
      f"({100 * (1 - abs(b_exc) / abs(b_all)):.0f}% attenuation)")
    a("Most of the association therefore arises outside the lesion territory.")

    text = "\n".join(L)
    out = ROOT / "lesion_excluded_regression_summary.txt"
    out.write_text(text + "\n")
    print(text)
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
