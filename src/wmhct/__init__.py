"""
wmhct - segmentation of white matter hypoattenuations in non-contrast CT.

Code for the paper: "AI-Based Pipeline for the Segmentation of White Matter Hypoattenuations in CT
Scans: A Design-Choice Validation Study".
See CITATION.cff for the citation.

Submodules follow the pipeline order:

    wmhct.data_prep         DICOM to NIfTI, BIDS sorting, FLAIR identification,
                            reference label pairing
    wmhct.preprocessing     HU clipping, brain windowing, skull stripping
    wmhct.registration      FLAIR to CT, masks to CT, CT to template
    wmhct.nnunet            dataset.json, cross-validation splits, resampling
    wmhct.back_projection   invert the transforms back into native FLAIR space
    wmhct.evaluation        measurement: masks and spreadsheets to numbers
    wmhct.statistics        inference: every p-value, correlation, effect size and limit
    wmhct.statistics.pvs    perivascular space analysis
    wmhct.tools             mask renaming for the Section 3.7 analysis

Paths and external tool locations are centralised in `wmhct.config` and read from
environment variables.
"""

__version__ = "1.0.0"

__all__ = [
    "config",
    "data_prep",
    "preprocessing",
    "registration",
    "nnunet",
    "back_projection",
    "evaluation",
    "statistics",
    "tools",
]
