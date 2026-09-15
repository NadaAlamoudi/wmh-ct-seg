"""
Stage 1.3 - Rule-based MRI sequence identification from DICOM header fields.
Manuscript: Methods (identification of FLAIR series)
"""

import pandas as pd


class MRIClassifier:
    def __init__(self):
        pass

    def _classify_mri_sequence(self, scanning_sequence, sequence_name, TR, TE, flip_angle, TI=None):
        if isinstance(scanning_sequence, str):
            if "GR" in scanning_sequence:
                if isinstance(sequence_name, str) and "swi" in sequence_name:
                    return "SWI"
                elif TE < 30 and 70 <= flip_angle <= 110 and (TI == 0 or pd.isna(TI)):
                    return "GRET1"
                elif TE < 30 and 5 <= flip_angle <= 20 and (TI == 0 or pd.isna(TI)):
                    return "GRET2"  # GRE T2* ('*' is not a valid filename character)
                elif len(scanning_sequence) == 2:  # GR is the only value in the field
                    return "GRE"

            # Inversion recovery sequences can be FLAIR (fluid nulled) or STIR (fat
            # nulled); the inversion time decides which.
            elif (TI is not None) and ((TI >= 1700) or ("IR" in scanning_sequence)):
                if TI is not None:
                    if TR >= 2000 and TE >= 60 and (flip_angle == 90 or flip_angle == 180) and 120 <= TI <= 170:
                        return "STIR"
                    elif TR >= 3000 and TE >= 80 and 1700 <= TI <= 2800:
                        return "FLAIR"

            # Spin echo sequences: T1, T2, PD or FSE, from the TE/TR/FA ranges.
            elif ("SE" in scanning_sequence or "RM" in scanning_sequence) and (TI == 0 or pd.isna(TI)):
                if "EP" in scanning_sequence and (isinstance(sequence_name, str) and "ep_b" in sequence_name):
                    return "DWI"
                elif TR < 800 and TE < 30 and flip_angle == 90:
                    return "T1W"
                elif TR > 1000 and TE < 30 and flip_angle == 90:
                    return "PD"
                elif TR > 2000 and TE > 80 and flip_angle == 90:
                    return "T2W"
                elif TR > 2000 and 80 >= TE > 50 and flip_angle == 90:
                    return "FSET2"
                else:
                    return "Unknown"

            # Echo planar sequences: diffusion, perfusion or functional.
            elif "EP" in scanning_sequence:
                if isinstance(sequence_name, str) and "ep_b" in sequence_name:
                    return "DWI"
                else:
                    return "Unknown_EP"

        return "Unknown"

    def classify(self, scanning_sequence, sequence_name, TR, TE, flip_angle, TI=None):
        return self._classify_mri_sequence(scanning_sequence, sequence_name, TR, TE, flip_angle, TI)
