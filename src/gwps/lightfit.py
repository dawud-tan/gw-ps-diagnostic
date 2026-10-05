"""Per-light intensity E, LED axis a and falloff mu from a matte (primer-white) ball of
known radius at several positions: centres from silhouettes, light position from the
mirror-ball triangulation.

Model (CLAUDE.md): I = rho * E * max(0, -l.a)^mu * max(0, n.l) / r^2. The primer albedo
rho is shared by every light, so the fitted E is rho_primer * E and PS albedo comes out
relative to the primer. What PS needs is that E * max(0, -l.a)^mu / r^2 is right across
the garment volume, so the fit is judged by leave-one-position-out prediction, not by
how well mu and a are pinned down individually (with mu ~ 1 they barely are: the
falloff changes the intensity by ~2 % over +-11 deg).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from .ps import NoiseModel
from .sphere import ray_sphere, sphere_roi


def ball_samples(lit, ambient, cam, C, radius, margin_px=2.0, sat_level=0.98):
    """Pixels on the ball at least margin_px inside the limb: S, n (camera frame), I, saturated."""
    r0, r1, c0, c1 = sphere_roi(cam, C, radius)
    jj, ii = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
    uv = np.stack([jj.ravel() + 0.5, ii.ravel() + 0.5], -1)
    rays = cam.rays(uv)
    D = np.linalg.norm(C)
    alpha = np.arcsin(radius / D)
    keep = np.arccos(np.clip(rays @ (C / D), -1, 1)) < alpha - margin_px / float(cam.fx)
    t = ray_sphere(rays[keep], C, radius)
    S = t[:, None] * rays[keep]
    n = (S - C) / radius
    raw = lit[r0:r1, c0:c1].ravel()[keep]
    I = raw - (ambient[r0:r1, c0:c1].ravel()[keep] if ambient is not None else 0)
    return {"S": S, "n": n, "I": I, "sat": raw >= sat_level}


def _basis(a0):
    ref = np.array([0, 1.0, 0]) if abs(a0[1]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(a0, ref)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a0, e1)


def _predict(x, S, n, P, a0, e1, e2):
    logE, al, be, mu = x[:4]
    Pk = P + x[4:7] if len(x) > 4 else P
    a = a0 + al * e1 + be * e2
    a /= np.linalg.norm(a)
    v = Pk - S
    r = np.linalg.norm(v, axis=1)
    l = v / r[:, None]
    aniso = np.clip(-(l @ a), 0, None) ** mu
    return np.exp(logE) * aniso * np.clip(np.einsum("ij,ij->i", n, l), 0, None) / r ** 2, a


@dataclass
class LightFitResult:
    E: float
    axis: np.ndarray
    mu: float
    sigma_E_rel: float
    sigma_mu: float
    rms_rel: float
    n_px: int
    lopo_rel: list | None = None
    position_shift_mm: np.ndarray | None = None


def _select(samples, P, noise, min_cos):
    S = np.concatenate([s["S"] for s in samples])
    n = np.concatenate([s["n"] for s in samples])
    I = np.concatenate([s["I"] for s in samples])
    sat = np.concatenate([s["sat"] for s in samples])
    pos = np.concatenate([np.full(len(s["I"]), k) for k, s in enumerate(samples)])
    l = P - S
    l /= np.linalg.norm(l, axis=1, keepdims=True)
    sig = np.sqrt(noise.var_total(I))
    ok = (~sat) & (np.einsum("ij,ij->i", n, l) >= min_cos) & (I > 3 * np.sqrt(noise.var(I)))
    return S[ok], n[ok], I[ok], sig[ok], pos[ok]


def fit_light(samples, P, noise=NoiseModel(), min_cos=0.2, fit_position=False, lopo=True):
    """samples: list (one per ball position) of ball_samples() dicts for this light."""
    S, n, I, sig, pos = _select(samples, P, noise, min_cos)
    a0 = S.mean(0) - P
    a0 /= np.linalg.norm(a0)
    e1, e2 = _basis(a0)

    def solve(Ss, ns, Is, sg, fit_pos):
        l = P - Ss
        r = np.linalg.norm(l, axis=1)
        cosn = np.einsum("ij,ij->i", ns, l / r[:, None])
        E0 = float(np.median(Is * r ** 2 / cosn))
        x0 = np.array([np.log(E0), 0, 0, 1.0] + ([0, 0, 0] if fit_pos else []))
        lo = [-np.inf, -1, -1, 0] + ([-0.05] * 3 if fit_pos else [])
        hi = [np.inf, 1, 1, 50] + ([0.05] * 3 if fit_pos else [])
        return least_squares(lambda x: (Is - _predict(x, Ss, ns, P, a0, e1, e2)[0]) / sg, x0,
                             bounds=(lo, hi), loss="soft_l1", f_scale=3.0, x_scale="jac")

    res = solve(S, n, I, sig, False)
    pred, a = _predict(res.x, S, n, P, a0, e1, e2)
    J = res.jac
    try:
        cov = np.linalg.inv(J.T @ J) * max(2 * res.cost / max(len(I) - 4, 1), 1.0)
        sE, smu = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[3, 3]))
    except np.linalg.LinAlgError:
        sE = smu = float("inf")
    out = LightFitResult(float(np.exp(res.x[0])), a, float(res.x[3]), sE, smu,
                         float(np.sqrt(np.mean(((I - pred) / pred) ** 2))), int(len(I)))
    if lopo and len(samples) >= 3:
        errs = []
        for k in range(len(samples)):
            tr, te = pos != k, pos == k
            if te.sum() < 50:
                continue
            r_k = solve(S[tr], n[tr], I[tr], sig[tr], False)
            p_k = _predict(r_k.x, S[te], n[te], P, a0, e1, e2)[0]
            errs.append(float(np.median(I[te] / p_k) - 1))
        out.lopo_rel = errs
    if fit_position:
        rp = solve(S, n, I, sig, True)
        out.position_shift_mm = rp.x[4:7] * 1000
    return out


def irradiance_scale(E, axis, mu, P, X):
    """|b| = E * max(0, -l.a)^mu / r^2 at points X (N,3): what PS actually uses per light."""
    v = P - X
    r = np.linalg.norm(v, axis=1)
    l = v / r[:, None]
    g = np.ones(len(X)) if axis is None or mu == 0 else np.clip(-(l @ axis), 0, None) ** mu
    return E * g / r ** 2
