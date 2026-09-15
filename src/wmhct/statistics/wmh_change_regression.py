"""
Section 3.6 - what predicts the WMH volume difference.

Manuscript: Section 3.6, "Previous stroke lesion volume and time between scans influence
    the WMH volume difference only in MRI measurements"

Run: wmhct-change-regression --help

Note: Pass --sheet For_statistical_analyses for the published 218-row sample.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro

OUTCOMES = ["WMHstable_vol", "WMHshrink_vol", "WMHgrow_vol",
            "WMHstable_count", "WMHshrink_count", "WMHgrow_count"]
PREDICTORS = ["Time_Difference_Days", "Modality", "Age", "Sex"]



# ---------------------------------------------------------------------------
# Table VIII
# ---------------------------------------------------------------------------
# Methods 2.8.3: "To identify factors associated with changes in WMH from V0 to V1
# within each group, we fitted two general linear models (one per group). The change in
# WMH served as the dependent variable, with baseline WMH volume as the main predictor,
# adjusting for age, sex, inter-scan interval, index stroke lesion volume, and old stroke
# lesion volume."
#
# This is a different specification from the pooled models above, which regress each of
# the six change outcomes on time, modality, age and sex. Both are kept: the pooled
# models are what the original OLS_for_all_variables.py fitted, and this is what the
# manuscript tabulates.
TABLE8_PREDICTORS = [
    ("WMH_vol_v0", "WMH Vol (V0)"),
    ("Age", "Age (V0)"),
    ("Sex", "Sex"),
    ("Time_Difference_Days", "Time (days)"),
    ("indexSLvol_mm3", "index SL Vol"),
    ("oldSLvol_mm3", "old SL Vol"),
]


def fit_table8(df, group_col, alpha, out_dir):
    """One general linear model per group, outcome = WMH volume change (V1 - V0)."""
    needed = ["WMH_vol_v0", "WMH_vol_v1"] + [c for c, _ in TABLE8_PREDICTORS]
    absent = [c for c in needed if c not in df.columns]
    if absent:
        print(f"\nTable VIII skipped: columns absent {absent}. The curated sheet "
              f"'For_statistical_analyses' carries all of them.")
        return None

    work = df.copy()
    work["WMH_change"] = work["WMH_vol_v1"] - work["WMH_vol_v0"]

    rows = []
    for group in sorted(work[group_col].dropna().unique()):
        sub = work[work[group_col] == group]
        cols = [c for c, _ in TABLE8_PREDICTORS]
        sub = sub[["WMH_change"] + cols].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < len(cols) + 2:
            print(f"Table VIII: {group} has too few complete rows ({len(sub)}).")
            continue
        X = sm.add_constant(sub[cols])
        model = sm.OLS(sub["WMH_change"], X).fit()
        for col, label in TABLE8_PREDICTORS:
            rows.append({
                "Group": group,
                "Predictor": label,
                "column": col,
                "B_value": float(model.params[col]),
                "Std_error": float(model.bse[col]),
                "p_value": float(model.pvalues[col]),
                "Significant": "Yes" if model.pvalues[col] < alpha else "No",
                "n": int(model.nobs),
                "R_squared": float(model.rsquared),
            })

    table8 = pd.DataFrame(rows)
    if table8.empty:
        return None
    table8.to_csv(out_dir / "table_viii_wmh_change_glm.csv", index=False)
    print("\nTable VIII - drivers of WMH change, one model per group "
          "(outcome: WMH_vol_v1 - WMH_vol_v0):")
    print(table8[["Group", "Predictor", "B_value", "Std_error", "p_value",
                  "Significant", "n"]].to_string(index=False,
                                                 float_format=lambda v: f"{v:.4f}"))
    return table8


def main():
    parser = argparse.ArgumentParser(
        description="Section 3.6: OLS of WMH change on time interval, modality, age and sex")
    parser.add_argument("--sheet", default=0,
                        help="worksheet name or index. The curated analysis table is "
                             "the sheet 'For_statistical_analyses' of "
                             "combined_data_for_analysis_and_paper1_MVH.xlsx (218 rows, "
                             "46 CT-MRI and 172 MRI-MRI), which is the sample the "
                             "manuscript reports. The default first sheet of that "
                             "workbook is All_measurements and is NOT that sample.")
    parser.add_argument("--combined_xlsx", required=True,
                        help="combined_data_for_analysis_corrected_allData.xlsx")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--group_col", default="Group")
    parser.add_argument("--id_col", default=None,
                        help="Participant identifier, for the independence check")
    parser.add_argument("--correction", choices=["none", "holm", "fdr_bh"], default="none",
                        help="Across the coefficient tests. 'none' reproduces the original")
    parser.add_argument("--log_outcome", action="store_true",
                        help="Fit on log1p(outcome), usually appropriate for skewed volumes")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--no_table8", action="store_true",
                        help="skip the per-group general linear models of Table VIII and "
                             "fit only the pooled models of the original script")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sheet = args.sheet
    if isinstance(sheet, str) and sheet.isdigit():
        sheet = int(sheet)
    df = pd.read_excel(args.combined_xlsx, sheet_name=sheet)

    if "Sex" in df.columns and df["Sex"].dtype == object:
        df["Sex"] = df["Sex"].map({"M": 0, "F": 1})
    df["Modality"] = df[args.group_col].map({"CT-MRI": 0, "MRI-MRI": 1})

    if df["Modality"].isna().any():
        bad = sorted(df.loc[df["Modality"].isna(), args.group_col].dropna().unique())
        raise ValueError(f"Unmapped values in '{args.group_col}': {bad}. "
                         f"Expected 'CT-MRI' and 'MRI-MRI'.")

    if args.id_col and args.id_col in df.columns:
        counts = df[args.id_col].value_counts()
        repeated = int((counts > 1).sum())
        if repeated:
            print(f"WARNING: {repeated} participant(s) contribute more than one row. OLS "
                  f"treats these as independent, so the standard errors below are too "
                  f"small. Consider mixedlm with a random intercept per participant.")
        else:
            print("Independence check: one row per participant.")
    else:
        print("NOTE: no --id_col supplied, so within-participant repetition was not "
              "checked. See the module docstring.")

    predictors = [p for p in PREDICTORS if p in df.columns]
    if set(predictors) != set(PREDICTORS):
        print(f"WARNING: predictors absent and dropped: {sorted(set(PREDICTORS) - set(predictors))}")

    outcomes = [c for c in OUTCOMES if c in df.columns]
    rows, diagnostics = [], []

    for var in outcomes:
        y_raw = pd.to_numeric(df[var], errors="coerce")
        y = np.log1p(y_raw) if args.log_outcome else y_raw
        y = y.dropna()

        X = sm.add_constant(df.loc[y.index, predictors].astype(float))
        keep = X.notna().all(axis=1)
        X, y = X[keep], y[keep]

        if len(y) <= len(predictors) + 2:
            print(f"Skipping {var}: only {len(y)} complete observations.")
            continue

        model = sm.OLS(y, X).fit()
        resid = model.resid

        shapiro_p = float(shapiro(resid).pvalue) if 3 <= len(resid) <= 5000 else np.nan
        try:
            bp_p = float(het_breuschpagan(resid, model.model.exog)[1])
        except Exception:
            bp_p = np.nan

        diagnostics.append({
            "Outcome": var, "n": int(len(y)),
            "outcome_transform": "log1p" if args.log_outcome else "identity",
            "R_squared": float(model.rsquared),
            "adj_R_squared": float(model.rsquared_adj),
            "F_pvalue": float(model.f_pvalue),
            "resid_shapiro_p": shapiro_p,
            "breusch_pagan_p": bp_p,
            "durbin_watson": float(durbin_watson(resid)),
            "condition_number": float(model.condition_number),
        })

        for name in ["const"] + predictors:
            rows.append({
                "Outcome": var, "Predictor": name,
                "Coefficient": float(model.params[name]),
                "StdErr": float(model.bse[name]),
                "CI_low": float(model.conf_int().loc[name, 0]),
                "CI_high": float(model.conf_int().loc[name, 1]),
                "p_raw": float(model.pvalues[name]),
                "R_squared": float(model.rsquared),
                "n": int(len(y)),
            })

    if not rows:
        print("No model could be fitted.")
        return

    results = pd.DataFrame(rows)
    diag = pd.DataFrame(diagnostics)

    # Correction across the coefficient tests, excluding the intercepts, which are not
    # hypotheses of interest.
    mask = results["Predictor"] != "const"
    results["p_adjusted"] = np.nan
    if args.correction != "none":
        adj = multipletests(results.loc[mask, "p_raw"].values,
                            method="holm" if args.correction == "holm" else "fdr_bh")[1]
        results.loc[mask, "p_adjusted"] = adj
    results["correction"] = args.correction
    decision = "p_adjusted" if args.correction != "none" else "p_raw"
    results["Significant"] = results[decision] < args.alpha
    results.loc[~mask, "Significant"] = np.nan

    results.to_csv(out_dir / "section_3_6_regression_coefficients.csv", index=False)
    diag.to_csv(out_dir / "section_3_6_regression_diagnostics.csv", index=False)

    with pd.ExcelWriter(out_dir / "section_3_6_regression_by_predictor.xlsx") as writer:
        for pred in predictors:
            results[results["Predictor"] == pred].to_excel(
                writer, sheet_name=f"{pred[:28]}", index=False)
        diag.to_excel(writer, sheet_name="diagnostics", index=False)

    print("\nCoefficients:")
    print(results.loc[mask, ["Outcome", "Predictor", "Coefficient", "p_raw",
                             "p_adjusted", "Significant"]].to_string(index=False))

    print("\nDiagnostics:")
    print(diag.to_string(index=False))

    bad_norm = diag[diag["resid_shapiro_p"] < 0.05]["Outcome"].tolist()
    bad_het = diag[diag["breusch_pagan_p"] < 0.05]["Outcome"].tolist()
    if bad_norm:
        print(f"\nWARNING: residuals depart from normality for {bad_norm}. OLS p-values "
              f"for these models are unreliable; try --log_outcome.")
    if bad_het:
        print(f"WARNING: heteroscedastic residuals for {bad_het}. Consider --log_outcome "
              f"or robust standard errors (model.fit(cov_type='HC3')).")

    # The section's headline claim, stated explicitly so it can be checked.
    for pred in ("Modality", "Time_Difference_Days"):
        if pred in predictors:
            sub = results[results["Predictor"] == pred]
            n_sig = int(sub["Significant"].fillna(False).sum())
            print(f"\n{pred}: significant for {n_sig} of {len(sub)} outcomes "
                  f"({args.correction} correction).")
    print("\nSection 3.6 claims modality matters and the time interval does not. Quote "
          "the correction you actually applied, and state it in the Methods.")

    if not args.no_table8:
        fit_table8(df, args.group_col, args.alpha, out_dir)


if __name__ == "__main__":
    main()
