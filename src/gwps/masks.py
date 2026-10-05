"""Object masks for turntable captures: what turns with the turntable (garment, mannequin,
turntable top) against what stays still in the image (backdrop, floor, the turntable's base).

Why: the camera and the studio stay still while the garment turns, so everything static looks
the same in every photo. COLMAP matches on it contradict the orbit, and Gaussian Wrapping would
model it as floaters. GW (11e3b6f) composites a training image's alpha onto its background
colour (train.py lines 291-293), and with --use_valid_mask (on in its radegs script) it treats
extraction points outside every view's mask as empty, so a mask must cover everything to be
meshed: garment, mannequin, turntable top.

Sources:
- 'silhouette' (default): one backlit frame per step (backdrop on, PS lights off). The object is
  dark against a bright backdrop whatever the garment's colour. Holes where the backdrop shows
  through (under an arm) stay background.
- 'sfm': threshold the developed SfM frame against a matte black backdrop. No extra frame, but a
  dark garment on a black backdrop fails. Holes are filled.
A static exclusion mask (white = static structure in front of the backdrop, such as the
turntable's base or a stand), painted once per rig, is removed from every mask.

Contract: <session>/masks/<colmap_image_name>, 8-bit PNG in the geometry of colmap_images/,
255 = object, 0 = background or excluded.
"""
from __future__ import annotations

import cv2
import numpy as np


def default_dilate_px(width):
    """Grow the object by ~0.1 % of the width (6 px at 6000 px) so its blurred edge stays in."""
    return max(1, int(round(0.001 * width)))


def default_feature_margin_px(width):
    """COLMAP feature masks are eroded by ~0.4 % of the width (24 px at 6000 px): a keypoint on
    the silhouette describes background that stays still while the garment turns."""
    return max(2, int(round(0.004 * width)))


def otsu(values, bins=256):
    """Otsu's threshold of a 1-D sample (between its 0.1 and 99.9 percentiles). Where two classes
    are separated by a gap, every threshold in the gap scores the same; the middle of that plateau
    is returned, not its first bin (which sits on the edge of the lower class and flips noise)."""
    lo, hi = np.percentile(values, [0.1, 99.9])
    if hi <= lo:
        return float(lo)
    h, edges = np.histogram(np.clip(values, lo, hi), bins, (lo, hi))
    c = 0.5 * (edges[1:] + edges[:-1])
    w0 = np.cumsum(h)
    w1 = w0[-1] - w0
    m0 = np.cumsum(h * c) / np.maximum(w0, 1)
    m1 = ((h * c).sum() - np.cumsum(h * c)) / np.maximum(w1, 1)
    between = (w0 * w1 * (m0 - m1) ** 2)[:-1]
    top = np.flatnonzero(between >= between.max() * (1 - 1e-3))
    return float(0.5 * (edges[1:][top[0]] + edges[1:][top[-1]]))


def _fill_holes(obj):
    """Background regions not connected to the image border become object."""
    bg = (~obj).astype(np.uint8)
    n, lab = cv2.connectedComponents(bg, connectivity=4)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    keep_bg = np.isin(lab, border[border > 0])
    return obj | (~keep_bg & (bg > 0))


def object_mask(img, dark_object, fill_holes=False, min_area_frac=2e-4, dilate_px=0, eps=1e-4):
    """Two-level segmentation of a linear image on log intensity (Otsu), cleaned up.
    dark_object: the object is darker than the background (backlit silhouette). Object
    components smaller than min_area_frac of the image (specks) are dropped. -> bool mask, info."""
    x = np.log10(np.maximum(np.asarray(img, np.float32), eps))
    t = otsu(x.ravel()[:: max(1, x.size // 2_000_000)])
    obj = x < t if dark_object else x > t
    k3 = np.ones((3, 3), np.uint8)
    obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_OPEN, k3)
    obj = cv2.morphologyEx(obj, cv2.MORPH_CLOSE, k3).astype(bool)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(obj.astype(np.uint8), connectivity=8)
    big = np.flatnonzero(stats[:, cv2.CC_STAT_AREA] >= min_area_frac * obj.size)
    obj = np.isin(lab, big[big > 0])
    if fill_holes:
        obj = _fill_holes(obj)
    if dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1, 2 * dilate_px + 1))
        obj = cv2.dilate(obj.astype(np.uint8), k).astype(bool)
    return obj, {"log10_threshold": t, "threshold": float(10 ** t), "object_fraction": float(obj.mean())}


def silhouette_mask(img_linear, dilate_px=None):
    """Object mask from a backlit frame (linear): dark object on a bright backdrop."""
    d = default_dilate_px(img_linear.shape[1]) if dilate_px is None else dilate_px
    return object_mask(img_linear, dark_object=True, fill_holes=False, dilate_px=d)


def sfm_mask(img_linear, dilate_px=None):
    """Object mask from the SfM frame against a matte black backdrop (linear): bright object."""
    d = default_dilate_px(img_linear.shape[1]) if dilate_px is None else dilate_px
    return object_mask(img_linear, dark_object=False, fill_holes=True, dilate_px=d)


def exclude(mask, static):
    """Remove static structure (static: bool, True = exclude) from an object mask."""
    if static is None:
        return mask
    if static.shape != mask.shape:
        raise ValueError(f"exclusion mask {static.shape} does not match the image {mask.shape}")
    return mask & ~static


def feature_mask(mask, margin_px):
    """COLMAP feature mask: the object eroded by margin_px (255 = extract features)."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
    return cv2.erode(mask.astype(np.uint8) * 255, k)


def read_mask(path):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError(f"cannot read mask {path}")
    return m > 127


def write_mask(path, mask):
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)
