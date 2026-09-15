"""Stage 6: evaluation.

Volume computation, agreement analysis (Pearson, Deming, Bland-Altman, CCC), WMH burden
strata, cross-validation fold composition, and the overlap metrics used throughout.
"""

from wmhct.evaluation.metrics import (  # noqa: F401
    absolute_volume_error,
    dice,
    iou,
    precision,
    sensitivity,
    volume_ml,
)
