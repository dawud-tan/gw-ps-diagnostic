"""Near-light photometric stereo, batched per pixel.

Model per light k: I_k = (rho n) . b_k(X), with b_k from gwps.lights (position, E, LED
axis and falloff, 1/r^2). Every pixel has its own light matrix, so this solves one
weighted 3x3 normal-equation system per pixel. Individual lights are rejected per
pixel when saturated, shadowed (shading far below the pixel's brightest light) or
inconsistent with the fit (predicted <= 0, or a large normalised residual).
The per-pixel `conf` is raw; stage 4 replaces it with `local_confidence`.

Noise versus model error: the weights use the sensor noise alone (optimal for noise). Outlier
tests and chi^2 use noise plus a relative model-error term `rel` (non-Lambertian fabric, light
calibration), because on real data a real sensor is so quiet that a few-percent model error
would otherwise read as a many-sigma outlier and throw good lights away.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass
class NoiseModel:
    """Variance of one ambient-subtracted linear value (fractions of full scale):
    noise = read^2 (x2 for the ambient frame) + I / full_well; total = noise + (rel * I)^2.

    k: the exposure factor of a value merged from an exposure bracket (gwps.io.merge_brackets),
    which is the frame's raw value divided by k (k = its shutter over the base shutter). Its
    variance is then read^2 / k^2 + I / (k full_well): a 4x longer frame has 1/16 of the read
    variance and 1/4 of the shot variance in base units. k = 1 is a single exposure. The model
    error term is relative, so it does not depend on k."""
    read: float = 0.005
    full_well: float = 10000.0
    ambient_subtracted: bool = True
    rel: float = 0.0

    def var(self, I, k=1.0):
        f = 2.0 if self.ambient_subtracted else 1.0
        return f * self.read ** 2 / k ** 2 + np.clip(I, 0, None) / (self.full_well * k)

    def var_total(self, I, k=1.0):
        return self.var(I, k) + (self.rel * np.clip(I, 0, None)) ** 2

    @classmethod
    def load(cls, path, ambient_subtracted=True, rel=None):
        """From a stage-C radiometry.json ({"noise": {"read": .., "full_well": ..}})."""
        d = json.loads(Path(path).read_text())
        d = d.get("noise", d)
        m = cls(float(d["read"]), float(d["full_well"]), ambient_subtracted, float(d.get("rel", 0.0)))
        if rel is not None:
            m.rel = float(rel)
        return m

    def to_dict(self):
        return asdict(self)


# Default relative model error. Measured on a synthetic sheen fabric with a quiet (real-sensor-like)
# noise model: larger values keep sheen-contaminated lights (median PS error 1.22 deg at 0,
# 1.46 at 0.005, 1.66 at 0.02), smaller ones would drop good lights over the ~1-2 % light-
# calibration error of a real rig. Relative confidence (local_confidence), not this term, is what
# keeps confidence from collapsing (38 % of pixels usable with absolute chi^2 and 0).
MODEL_ERROR = 0.005


def make_noise_model(noise=None, read_noise=0.005, full_well=10000.0, model_error=MODEL_ERROR):
    """A NoiseModel from an object, a radiometry.json path, or explicit read noise / full well."""
    if isinstance(noise, NoiseModel):
        return noise
    if noise:
        return NoiseModel.load(noise, rel=model_error)
    return NoiseModel(read_noise, full_well, True, model_error)


@dataclass
class PSParams:
    shadow_frac: float = 0.1       # reject light if its shading < shadow_frac * brightest shading
    min_signal_sigma: float = 3.0  # and if I < this many noise sigmas
    outlier_sigma: float = 5.0     # drop worst light if |normalised residual| exceeds this
    outlier_iters: int = 2
    min_lights: int = 3            # 3 = solve but flag low confidence


def light_vectors(lights, X):
    """lights: list of gwps.lights.Light in the camera frame; X (N,3) -> b (K,N,3)."""
    return np.stack([l.light_vector(X)[0] for l in lights])


def _solve(b, I, w):
    A = np.einsum("kn,kni,knj->nij", w, b, b)
    y = np.einsum("kn,kn,kni->ni", w, I, b)
    det = np.linalg.det(A)
    good = np.abs(det) > 1e-12 * np.maximum(np.einsum("nii->n", A), 1e-30) ** 3
    g = np.zeros_like(y)
    if good.any():
        g[good] = np.linalg.solve(A[good], y[good, :, None])[..., 0]
    return g, good


def solve(I, sat, b, noise=NoiseModel(), params=PSParams(), use_init=None, k=1.0):
    """I, sat: (K,N) ambient-subtracted intensities and saturation flags; b: (K,N,3).
    use_init: optional (K,N) initial light selection (else chosen from the intensities).
    k: (K,N) exposure factors of bracket-merged values (NoiseModel), or 1.0.

    Returns dict of n (N,3), albedo (N,), conf (N,), n_lights (N,), ok (N,), low_conf (N,),
    chi2 and dof (for local_confidence).
    """
    I = np.asarray(I, np.float64)
    K, N = I.shape
    bn = np.linalg.norm(b, axis=2)
    sigma = np.sqrt(noise.var(I, k))
    sigma_t = np.sqrt(noise.var_total(I, k))
    if use_init is not None:
        use = use_init & (~sat) & (bn > 0)
    else:
        shading = np.where(bn > 0, I / np.where(bn > 0, bn, 1), 0)
        use = (~sat) & (bn > 0) & (I > params.min_signal_sigma * sigma)
        use &= shading >= params.shadow_frac * np.max(np.where(use, shading, 0), axis=0, keepdims=True)

    w_all = 1.0 / sigma ** 2
    for it in range(params.outlier_iters + 1):
        w = np.where(use, w_all, 0.0)
        g, good = _solve(b, I, w)
        pred = np.einsum("kni,ni->kn", b, g)
        r = (I - pred) / sigma_t
        n_used = use.sum(0)
        if it == params.outlier_iters:
            break
        # lights that the fit says face away, or fit badly: drop the worst one per pixel
        bad = np.where(use, np.abs(r) + np.where(pred <= 0, 1e6, 0), -1)
        worst = np.argmax(bad, axis=0)
        worst_val = bad[worst, np.arange(N)]
        drop = (worst_val > params.outlier_sigma) & (n_used > params.min_lights)
        drop |= (worst_val >= 1e6) & (n_used > 0)
        if not drop.any():
            break
        use[worst[drop], np.flatnonzero(drop)] = False

    n_used = use.sum(0)
    albedo = np.linalg.norm(g, axis=1)
    n = g / np.where(albedo > 0, albedo, 1)[:, None]
    ok = good & (n_used >= params.min_lights) & (albedo > 0)
    chi2 = np.where(use, r ** 2, 0).sum(0)
    dof = n_used - 3
    red = np.where(dof > 0, chi2 / np.maximum(dof, 1), 1.0)
    conf = np.where(dof > 0, 1.0 / np.maximum(red, 1.0), 0.25)
    conf = np.where(ok, conf, 0.0)
    return {"n": n, "albedo": albedo, "conf": conf, "n_lights": n_used.astype(np.int8),
            "ok": ok, "low_conf": ok & (n_used == 3), "chi2": np.where(ok, chi2, 0.0),
            "dof": np.where(ok, np.maximum(dof, 0), 0), "use": use}


def local_confidence(chi2, dof, ok, n_lights, sigma_px=2.0, relative=True):
    """Confidence from the reduced chi^2 pooled over a small neighbourhood (images, H x W).

    A single pixel's chi^2_red with 5 dof has std ~0.6, so a per-pixel confidence is itself
    noise; used as a low-pass weight it unbalances pixel-scale texture into the low band.
    Pooling keeps real model failures (sheen, interreflection: many pixels) and drops noise.

    relative=True divides by the view's median pooled chi^2_red, so confidence marks regions
    that fit worse than is typical for this view. On real data the absolute chi^2 scale rests
    on the noise and model-error estimates; a global mismatch would otherwise drive every
    pixel below min_conf. The median itself is reported by stage 4.
    Returns (conf image, median pooled chi^2_red).
    """
    import cv2
    k = int(2 * np.ceil(3 * sigma_px) + 1)
    blur = lambda x: cv2.GaussianBlur(x.astype(np.float32), (k, k), sigma_px, borderType=cv2.BORDER_CONSTANT)
    m = ok & (dof > 0)
    red = blur(np.where(m, chi2, 0)) / np.maximum(blur(np.where(m, dof, 0)), 1e-6)
    med = float(np.median(red[m])) if m.any() else 1.0
    scale = max(med, 1.0) if relative else 1.0
    conf = 1.0 / np.maximum(red / scale, 1.0)
    conf = np.where(n_lights == 3, 0.25, conf)
    return np.where(ok, conf, 0.0).astype(np.float32), med


def solve_view(I, sat, b, hit, noise=NoiseModel(), params=PSParams(), smooth_px=3.0, chunk=400_000, k=None):
    """Two-pass solve for one view. I, sat: (K,N) over the hit pixels of the (H,W) mask `hit`
    (row-major); b: (K,N,3) array, or a function idx -> (K,len(idx),3) so that the light
    vectors of a 24 MP view are never all in memory at once (processed in chunks).
    k: optional (K,N) exposure factors of bracket-merged values (NoiseModel); None = 1.

    Choosing lights per pixel from that pixel's own noisy intensity is a selection bias: a
    light near the threshold is kept exactly where noise pushed it up, which tilts the normal
    toward it (measured: 0.11 deg uniform bias on a board at 50 deg tilt, where one light was
    kept in ~44 % of pixels). Pass 2 therefore selects lights from the *predicted* shading of
    a spatially smoothed pass-1 normal (attached shadow / grazing), and detects cast shadows
    per pixel only by a large negative residual (observed far below the pass-1 prediction).
    """
    import cv2
    K, N = I.shape
    b_fn = b if callable(b) else (lambda idx: b[:, idx])
    k_of = (lambda idx: 1.0) if k is None else (lambda idx: k[:, idx])
    chunks = [np.arange(s, min(s + chunk, N)) for s in range(0, N, chunk)]
    n1 = np.zeros((N, 3), np.float32)
    alb1 = np.zeros(N, np.float32)
    ok1 = np.zeros(N, bool)
    for idx in chunks:
        r = solve(I[:, idx], sat[:, idx], b_fn(idx), noise, params, k=k_of(idx))
        n1[idx], alb1[idx], ok1[idx] = r["n"], r["albedo"], r["ok"]
    H, W = hit.shape
    okimg = np.zeros((H, W), np.float32)
    okimg[hit] = ok1
    n_img = np.zeros((H, W, 3), np.float32)
    n_img[hit] = n1 * ok1[:, None]
    ksize = int(2 * np.ceil(3 * smooth_px) + 1)
    blur = lambda x: cv2.GaussianBlur(x, (ksize, ksize), smooth_px, borderType=cv2.BORDER_CONSTANT)
    ns = blur(n_img) / np.maximum(blur(okimg), 1e-6)[..., None]
    del n_img, okimg
    ns = ns[hit]
    ns /= np.maximum(np.linalg.norm(ns, axis=1, keepdims=True), 1e-12)
    keys = ("n", "albedo", "conf", "n_lights", "ok", "low_conf", "chi2", "dof")
    out = {"n": np.zeros((N, 3), np.float32), "albedo": np.zeros(N, np.float32), "conf": np.zeros(N, np.float32),
           "n_lights": np.zeros(N, np.int8), "ok": np.zeros(N, bool), "low_conf": np.zeros(N, bool),
           "chi2": np.zeros(N, np.float32), "dof": np.zeros(N, np.int16)}
    for idx in chunks:
        bb = b_fn(idx)
        Ic = np.asarray(I[:, idx], np.float64)
        pred = np.einsum("kni,ni->kn", bb, ns[idx].astype(np.float64))
        use = (pred > 0) & (pred >= params.shadow_frac * pred.max(0, keepdims=True))
        g1 = n1[idx].astype(np.float64) * alb1[idx, None]
        kc = k_of(idx)
        res = (Ic - np.einsum("kni,ni->kn", bb, g1)) / np.sqrt(noise.var_total(Ic, kc))
        use &= ~(ok1[idx][None] & (res < -params.outlier_sigma))
        r2 = solve(Ic, sat[:, idx], bb, noise, params, use_init=use, k=kc)
        for key in keys:
            out[key][idx] = r2[key]
    return out
