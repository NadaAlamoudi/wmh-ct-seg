"""Stage 3: registration.

FLAIR to CT registration with reference-mask propagation (FSL FLIRT or NiftyReg
back-ends), and the CT-to-template arm used for the template-space experiment.

Settings as run:
    FLAIR to CT      FLIRT -dof 6 -cost normmi, or NiftyReg reg_aladin
    mask resampling  FLIRT -interp nearestneighbour, or reg_resample -inter 0
    CT to template   FLIRT -dof 12 (affine)
"""
