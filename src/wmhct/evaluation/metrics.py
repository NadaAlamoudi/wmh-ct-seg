"""
Overlap and volume metrics.
"""

import numpy as np


def _binarise(mask):
    return np.asarray(mask) > 0


def dice(prediction, reference):
    """Dice similarity coefficient. Returns 1.0 when both masks are empty."""
    p, r = _binarise(prediction), _binarise(reference)
    denom = p.sum() + r.sum()
    if denom == 0:
        return 1.0
    return float(2 * np.logical_and(p, r).sum() / denom)


def iou(prediction, reference):
    """Intersection over union. Returns 1.0 when both masks are empty."""
    p, r = _binarise(prediction), _binarise(reference)
    union = np.logical_or(p, r).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(p, r).sum() / union)


def sensitivity(prediction, reference):
    """True positive rate, also called recall. Undefined for an empty reference."""
    p, r = _binarise(prediction), _binarise(reference)
    if r.sum() == 0:
        return float('nan')
    return float(np.logical_and(p, r).sum() / r.sum())


def precision(prediction, reference):
    """Positive predictive value. Undefined for an empty prediction."""
    p, r = _binarise(prediction), _binarise(reference)
    if p.sum() == 0:
        return float('nan')
    return float(np.logical_and(p, r).sum() / p.sum())


def volume_ml(mask, zooms):
    """
    Lesion volume in millilitres.

    Parameters
    ----------
    mask : array-like
        Binary or label mask.
    zooms : sequence of float
        Voxel dimensions in mm, e.g. `nibabel_image.header.get_zooms()`.
    """
    voxel_mm3 = float(np.prod(np.asarray(zooms)[:3]))
    return float(_binarise(mask).sum() * voxel_mm3 / 1000.0)


def absolute_volume_error(prediction, reference, zooms):
    """Absolute difference between predicted and reference volume, in mL."""
    return abs(volume_ml(prediction, zooms) - volume_ml(reference, zooms))


def relative_volume_error(prediction, reference, zooms):
    """ Absolute volume error as a fraction of the reference volume."""
    ref = volume_ml(reference, zooms)
    if ref == 0:
        return float('nan')
    return absolute_volume_error(prediction, reference, zooms) / ref
