"""Radiometric checks of the real camera and lights, from the pilot's frames.

- Noise model (photon transfer): temporal variance of dark frames gives read noise; of a
  stack of flat frames (evenly lit card, locked exposure) gives variance versus mean over the
  card's spatial spread of levels; var = read^2 + mean / full_well. Units: fractions of full
  scale, as everywhere else. Replaces the synthetic defaults in stage 4.
- Linearity: median signal of the flat card over an exposure-time sweep; a line through the
  origin must fit (R^2 > 0.999, max deviation < 1 %) below the saturation onset.
- Additivity: the all-lights frame equals the sum of single-light frames (ambient
  subtracted); tests linearity in exactly the regime PS uses, and that lights don't interact.
- Drift: a light's frame repeated at the end of the session matches the start (< 1 %).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class NoiseFit:
    read: float
    full_well: float
    black_offset: float
    n_bins: int
    fit_rms_rel: float
    read_from: str
    dark_clipped_fraction: float


def noise_from_stacks(dark, flat, n_bins=24, roi=None):
    """dark, flat: (N,H,W) linear frames (N >= 2), on the same loading path PS uses.

    Fits var = read^2 + mean / full_well to the flat stack's temporal variance, binned over
    the card's spatial spread of levels. Dark frames give read^2 directly only when they are
    not clipped: RAW decoders subtract the black level and clip at 0, which leaves ~0.34 of
    the true variance (seen: 0.0027 for a true 0.005). Otherwise the fitted intercept is used.
    """
    dark = np.asarray(dark, np.float64)
    flat = np.asarray(flat, np.float64)
    if roi is not None:
        dark, flat = dark[:, roi[0]:roi[1], roi[2]:roi[3]], flat[:, roi[0]:roi[1], roi[2]:roi[3]]
    clipped = float(np.mean(dark <= 0))
    offset = float(np.median(dark.mean(axis=0))) if clipped < 0.01 else 0.0
    mu = flat.mean(axis=0) - offset
    var = flat.var(axis=0, ddof=1)
    good = (mu > 0.02) & (mu < 0.9)
    mu, var = mu[good], var[good]
    edges = np.quantile(mu, np.linspace(0, 1, n_bins + 1))
    xs, ys = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (mu >= a) & (mu < b)
        if m.sum() > 50:
            # means, not medians: a sample variance from N frames is chi^2_(N-1)-distributed and its
            # median is ~7 % low at N = 10, which would inflate the full well by the same amount
            xs.append(float(np.mean(mu[m])))
            ys.append(float(np.mean(var[m])))
    xs, ys = np.array(xs), np.array(ys)
    A = np.c_[np.ones_like(xs), xs]
    (a_fit, slope), *_ = np.linalg.lstsq(A, ys, rcond=None)
    if clipped < 0.01:
        read2, source = float(np.mean(dark.var(axis=0, ddof=1))), "dark frames"
        slope = float(np.sum((ys - read2) * xs) / np.sum(xs ** 2))
    else:
        read2, source = max(float(a_fit), 0.0), "flat-stack intercept (dark frames clipped)"
    pred = read2 + slope * xs
    rel = float(np.sqrt(np.mean(((ys - pred) / pred) ** 2)))
    return NoiseFit(float(np.sqrt(read2)), float(1.0 / max(slope, 1e-12)), offset, len(xs), rel, source, clipped)


def linearity(exposures, signals, sat_frac=0.9):
    """exposures (s), signals (median linear value). Fit s = k t through the origin on points
    below sat_frac of the maximum; report R^2, max relative deviation and saturation onset."""
    t = np.asarray(exposures, float)
    y = np.asarray(signals, float)
    o = np.argsort(t)
    t, y = t[o], y[o]
    use = y < sat_frac * min(1.0, y.max() if y.max() > 0 else 1.0)
    if use.sum() < 3:
        use = np.ones_like(y, bool)
    k = float(np.sum(t[use] * y[use]) / np.sum(t[use] ** 2))
    pred = k * t
    ss_res = float(np.sum((y[use] - pred[use]) ** 2))
    ss_tot = float(np.sum((y[use] - y[use].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    dev = (y - pred) / np.where(pred > 0, pred, 1)
    onset = next((float(y[i]) for i in range(len(y)) if dev[i] < -0.02 and not use[i]), None)
    return {"gain_per_s": k, "r2": float(r2), "max_rel_dev_linear_range": float(np.abs(dev[use]).max()),
            "saturation_onset_signal": onset, "points": [(float(a), float(b)) for a, b in zip(t, y)]}


def _signal_mask(img, lo=0.1, hi=0.9):
    m = img.max()
    return (img > lo * m) & (img < hi)


def additivity(all_img, singles, ambient=None, mask=None):
    """Median of (all - ambient) / sum(single - ambient) over well-exposed pixels."""
    amb = ambient if ambient is not None else 0.0
    total = np.sum([s - amb for s in singles], axis=0)
    a = all_img - amb
    m = _signal_mask(all_img) if mask is None else mask
    m &= total > 0
    return float(np.median(a[m] / total[m]))


def drift(start, end, ambient=None, mask=None):
    """Median of end / start over well-exposed pixels (ambient subtracted if given)."""
    amb = ambient if ambient is not None else 0.0
    s, e = start - amb, end - amb
    m = _signal_mask(start) if mask is None else mask
    m &= s > 0
    return float(np.median(e[m] / s[m]))


def to_dict(fit):
    return asdict(fit)
