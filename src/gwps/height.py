"""Implied height error of a verdict cluster, in mm: the low-passed normal difference integrated
over the surface (CLAUDE.md step 5, optional; Frankot-Chellappa was the suggested method).

If the real surface is the mesh moved by h along its normal, then n_real ~ n_mesh - grad_S h, so
grad_S h = n_mesh - n_ps (PS sees the real surface). Per view, in the camera frame, with the same
valid pixels, confidence weights, per-view rotation and vector low-pass at the cluster's scale as
stage 5, each step between two grid pixels is the line integral of (n_mesh - n_ps) along the
mesh's own 3-D positions there, h(b) - h(a) = g . (X_b - X_a): no camera model is needed. The
steps are solved for h by masked least squares (a Poisson equation on the valid pixels only),
which needs no values outside the garment; Frankot-Chellappa (also here, for cross-checking)
integrates over the whole rectangle and has the gradients zero-filled outside it. On an
irregular region with a hole (tests/test_I_height.py) that was off by 0.32 on a 0.5-high bump,
where the masked least squares were exact (4e-7); inside a cluster's crop, which is mostly valid,
the two agreed within 0.01 mm.

Only views within 45 deg of face-on (median over the cluster's pixels) are used. Beyond that the
image-space low-pass of the normals is no longer the gradient of one surface: on the synthetic
bump seen at 60 deg the field's curl was 8 % of its steps (0.7 % face-on), and the two
integrators then disagreed by 1.6 mm (3.9 against 2.3 mm), where face-on they agreed within
0.01 mm. With 36 turntable steps every face has such views.

h > 0: the surface PS sees lies outside the mesh there (the mesh is low, e.g. a fold GW smoothed
away); h < 0: the mesh bulges out. Reported relative to a ring around the cluster, over the
cluster with its holes filled (a bump's cluster is the ring of its flanks: its apex does not
tilt), as the value of largest magnitude (1st or 99th percentile, by the sign of the median) and
the median. Being the s-low-passed difference, features near s read smaller than they are: a
Gaussian bump of width sigma reads h sigma^2 / (sigma^2 + s^2) at its apex (a 3 mm, 10 mm bump
2.4 mm at s = 5 mm, 1.5 mm at s = 10 mm).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.ndimage import binary_dilation, binary_fill_holes
from scipy.sparse.linalg import spsolve

from .compare import CompareParams, apodise, kabsch, lowpass_vec, valid_mask

MAX_GRID_PX = 250_000      # integration grid per view and cluster; coarser strides beyond this


def runs_ok(sup, f, axis):
    """True at index j where sup holds at every pixel j..j+f along axis (a grid step of stride f
    that crosses no invalid pixel); False where j + f is past the edge."""
    c = np.cumsum((~sup).astype(np.int32), axis=axis)
    c = np.concatenate([np.zeros_like(np.take(c, [0], axis=axis)), c], axis=axis)
    n = sup.shape[axis]
    out = np.zeros(sup.shape, bool)
    if f < n:
        win = np.take(c, np.arange(f + 1, n + 1), axis=axis) - np.take(c, np.arange(0, n - f), axis=axis)
        sl = [slice(None)] * sup.ndim
        sl[axis] = slice(0, n - f)
        out[tuple(sl)] = win == 0
    return out


def integrate_masked(steps_u, steps_v, ok_u, ok_v, support):
    """Least-squares h on the support from steps along u (columns) and v (rows): steps_u[i, j] =
    h[i, j+1] - h[i, j], valid where ok_u; likewise v. One connected component's mean is free;
    a tiny ridge (1e-9 of the diagonal) fixes it. Returns h (NaN outside the support)."""
    H, W = support.shape
    idx = -np.ones((H, W), np.int64)
    n = int(support.sum())
    idx[support] = np.arange(n)
    rows, cols, vals, rhs = [], [], [], []
    e = 0
    for ok, steps, (di, dj) in ((ok_u, steps_u, (0, 1)), (ok_v, steps_v, (1, 0))):
        ii, jj = np.nonzero(ok)
        a, b = idx[ii, jj], idx[ii + di, jj + dj]
        keep = (a >= 0) & (b >= 0)
        a, b, s = a[keep], b[keep], steps[ii[keep], jj[keep]]
        m = len(a)
        r = np.arange(e, e + m)
        rows += [r, r]
        cols += [b, a]
        vals += [np.ones(m), -np.ones(m)]
        rhs.append(s)
        e += m
    h = np.full((H, W), np.nan)
    if e == 0 or n == 0:
        return h
    A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(e, n))
    N = (A.T @ A).tocsc()
    N = N + sparse.identity(n, format="csc") * (1e-9 * max(float(N.diagonal().max()), 1.0))
    h[support] = spsolve(N, A.T @ np.concatenate(rhs))
    return h


def frankot_chellappa(du, dv):
    """Frankot & Chellappa (1988): the integrable surface nearest to the gradients (forward
    differences du, dv) on the whole rectangle, periodic. Kept for comparison only: a garment's
    valid region is irregular, and the zero-filled gradients outside it are part of the fit."""
    H, W = du.shape
    wu = 2 * np.pi * np.fft.fftfreq(W)[None, :]
    wv = 2 * np.pi * np.fft.fftfreq(H)[:, None]
    eu, ev = np.exp(1j * wu) - 1, np.exp(1j * wv) - 1          # forward-difference operators
    den = np.abs(eu) ** 2 + np.abs(ev) ** 2
    den[0, 0] = 1.0
    Hf = (np.conj(eu) * np.fft.fft2(du) + np.conj(ev) * np.fft.fft2(dv)) / den
    Hf[0, 0] = 0.0
    return np.real(np.fft.ifft2(Hf))


MAX_VIEW_DEG = 45.0        # views of a cluster used for its height (see the module docstring)


def cluster_height_view(mesh_maps, ps_out, garment, cluster_faces, fx, s_mm, params: CompareParams,
                        ring_sigmas=2.0, min_px=30, method="lsq", max_view_deg=MAX_VIEW_DEG):
    """One view's implied height of one cluster (a boolean per mesh face), in metres:
    {"peak_m", "median_m", "px", "view_deg"} or None where the view sees too little of it or of
    its ring, or sees it more obliquely than max_view_deg."""
    valid, cos_view = valid_mask(mesh_maps, ps_out, garment, params)
    fid = mesh_maps["face_id"]
    cl = valid & cluster_faces[np.where(mesh_maps["hit"], fid, 0)] & mesh_maps["hit"]
    if cl.sum() < min_px:
        return None
    view_deg = float(np.degrees(np.arccos(np.clip(np.median(cos_view[cl]), -1, 1))))
    if view_deg > max_view_deg:
        return None
    w = apodise(valid, params.taper_px) * np.where(valid, ps_out["conf"], 0.0)
    n_mesh, n_ps = mesh_maps["normal_cam"], ps_out["n"]
    if params.remove_rotation and valid.sum() >= 3:                  # as stage 5 does
        Rk, _ = kabsch(n_ps[valid], n_mesh[valid], w[valid])
        n_ps = (n_ps @ Rk.T.astype(np.float32)).astype(np.float32)
    z_med = float(np.median(mesh_maps["depth"][valid]))
    sig = fx * (s_mm / 1000.0) / z_med
    # crop: the cluster plus its ring plus the low-pass's reach
    rows, cols = np.nonzero(cl)
    m = int(np.ceil(4 * sig + ring_sigmas * sig + 4))
    H, W = valid.shape
    r0, r1 = max(rows.min() - m, 0), min(rows.max() + m + 1, H)
    c0, c1 = max(cols.min() - m, 0), min(cols.max() + m + 1, W)
    sl = (slice(r0, r1), slice(c0, c1))
    lp_ps, den = lowpass_vec(np.ascontiguousarray(n_ps[sl]), np.ascontiguousarray(w[sl]), sig)
    lp_mesh, _ = lowpass_vec(np.ascontiguousarray(n_mesh[sl]), np.ascontiguousarray(w[sl]), sig)
    g = (lp_mesh - lp_ps).astype(np.float64)
    del lp_ps, lp_mesh
    X = mesh_maps["pos_cam"][sl].astype(np.float64)
    sup_full = valid[sl] & (den > params.min_weight)
    # stride: the low-passed field is smooth over sig, so a grid of sig/2 loses nothing
    f = max(1, min(int(sig // 2), int(np.ceil(np.sqrt(sup_full.size / MAX_GRID_PX)))))
    sub = (slice(None, None, f), slice(None, None, f))
    # a grid step must run over valid pixels the whole way (no hidden discontinuity in between)
    run_u, run_v = runs_ok(sup_full, f, 1), runs_ok(sup_full, f, 0)
    gs, Xs, sup = g[sub], X[sub], sup_full[sub]
    ok_u = np.zeros_like(sup)
    ok_v = np.zeros_like(sup)
    ok_u[:, :-1] = sup[:, :-1] & sup[:, 1:] & run_u[sub][:, :-1]
    ok_v[:-1, :] = sup[:-1, :] & sup[1:, :] & run_v[sub][:-1, :]
    du = np.zeros(sup.shape)
    dv = np.zeros(sup.shape)
    du[:, :-1] = np.einsum("ijk,ijk->ij", 0.5 * (gs[:, :-1] + gs[:, 1:]), Xs[:, 1:] - Xs[:, :-1])
    dv[:-1, :] = np.einsum("ijk,ijk->ij", 0.5 * (gs[:-1, :] + gs[1:, :]), Xs[1:, :] - Xs[:-1, :])
    if method == "fc":
        h = frankot_chellappa(np.where(ok_u, du, 0.0), np.where(ok_v, dv, 0.0))
    else:
        h = integrate_masked(du, dv, ok_u, ok_v, sup)
    clf = binary_fill_holes(cl[sl])[sub] & sup
    r = max(1, int(round(ring_sigmas * sig / f)))
    ring = binary_dilation(clf, iterations=r) & ~clf & sup & np.isfinite(h)
    if clf.sum() < 4 or ring.sum() < 4:
        return None
    dev = h[clf] - float(np.median(h[ring]))
    dev = dev[np.isfinite(dev)]
    med = float(np.median(dev))
    peak = float(np.percentile(dev, 99 if med >= 0 else 1))
    return {"peak_m": peak, "median_m": med, "px": int(cl.sum()), "grid_stride_px": f, "view_deg": view_deg}


def cluster_heights(views, garment, clusters, params: CompareParams, min_views=1, method="lsq"):
    """views: iterable of (mesh maps, ps_out, fx); clusters: [(key, faces, s_mm)]. -> {key:
    {"peak_mm", "median_mm", "views", "per_view": [...]}} (medians over the views that see it)."""
    F = len(garment)
    masks = {}
    for key, faces, s in clusters:
        m = np.zeros(F, bool)
        m[faces] = True
        masks[key] = (m, s)
    per = {key: [] for key in masks}
    for maps, ps_out, fx in views:
        for key, (m, s) in masks.items():
            r = cluster_height_view(maps, ps_out, garment, m, fx, s, params, method=method)
            if r is not None:
                per[key].append(r)
    out = {}
    for key, rs in per.items():
        if len(rs) < min_views:
            out[key] = None
            continue
        out[key] = {"peak_mm": 1000 * float(np.median([r["peak_m"] for r in rs])),
                    "median_mm": 1000 * float(np.median([r["median_m"] for r in rs])),
                    "views": len(rs), "per_view_peak_mm": [round(1000 * r["peak_m"], 3) for r in rs]}
    return out
