"""Spheres for calibration: centre from a backlit silhouette, mirror-ball highlights,
reflected rays and light triangulation. All pixel coordinates are COLMAP convention
(pixel (row i, col j) has its centre at (j + 0.5, i + 0.5)); `cam` is any camera with
rays(uv) -> unit camera-frame directions (PINHOLE or OPENCV).

Sensitivity to keep in mind: a lateral error dC in the mirror-ball centre tilts every
reflected ray by ~2 dC / R, i.e. moves a light ~2 L dC / R (L = ball-to-light distance).
With R = 40 mm and L = 1.25 m that is 6 mm of light position per 0.1 mm of centre.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares


def _sample(img, u, v):
    """Bilinear samples at COLMAP coordinates (u, v)."""
    return map_coordinates(img, [np.asarray(v) - 0.5, np.asarray(u) - 0.5], order=1, mode="nearest")


def find_dark_blob(img, r_min_px, r_max_px):
    """Initial (u0, v0, r0) of a dark disc (ball on a bright backdrop): the largest
    inscribed circle of the dark region (a stand is always thinner than the ball), accepted
    only if the ring just outside it is mostly bright (a stand covers a small sector)."""
    img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, mask = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    n, lab = cv2.connectedComponents(mask)
    dark = mask > 0
    best, score = None, 0.0
    for i in range(1, n):
        comp = lab == i
        k = np.argmax(np.where(comp, dist, 0))
        r = float(dist.flat[k])
        if not (r_min_px <= r <= r_max_px):
            continue
        ci, cj = np.unravel_index(k, dist.shape)
        ang = np.linspace(0, 2 * np.pi, 180, endpoint=False)
        ring = []
        for rr in (r + 3, r + 6):
            ii = np.clip(np.round(ci + rr * np.sin(ang)).astype(int), 0, img.shape[0] - 1)
            jj = np.clip(np.round(cj + rr * np.cos(ang)).astype(int), 0, img.shape[1] - 1)
            ring.append(~dark[ii, jj])
        frac = float(np.mean(ring))
        if frac > score:
            best, score = (float(cj + 0.5), float(ci + 0.5), r), frac
    if best is None or score < 0.7:
        raise RuntimeError("no disc-shaped dark blob found; is the silhouette frame backlit?")
    return best


def silhouette_edges(img, u0, v0, r0, step=0.1):
    """Sub-pixel edge points along radial profiles: the crossing of the local mid-level
    between the levels just inside and just outside the edge (unbiased for a symmetric
    blur, and insensitive to slow shading of ball or backdrop)."""
    hw = max(6.0, 0.15 * r0)
    n_ang = int(max(360, 2 * np.pi * r0))
    ang = 2 * np.pi * np.arange(n_ang) / n_ang
    rr = np.arange(r0 - hw, r0 + hw + step / 2, step)
    U = u0 + np.cos(ang)[:, None] * rr[None]
    V = v0 + np.sin(ang)[:, None] * rr[None]
    P = _sample(img.astype(np.float64), U.ravel(), V.ravel()).reshape(U.shape)
    inner = np.median(P[:, rr < r0 - hw / 2], axis=1)
    outer = np.median(P[:, rr > r0 + hw / 2], axis=1)
    mid = 0.5 * (inner + outer)
    contrast = outer - inner
    d = P - mid[:, None]
    cross = (d[:, :-1] * d[:, 1:] <= 0) & (d[:, :-1] != d[:, 1:])
    pts, ok = [], []
    for a in range(n_ang):
        idx = np.flatnonzero(cross[a])
        if idx.size == 0:
            ok.append(False)
            pts.append((np.nan, np.nan))
            continue
        i = idx[np.argmin(np.abs(rr[idx] - r0))]
        f = d[a, i] / (d[a, i] - d[a, i + 1])
        r = rr[i] + f * step
        pts.append((u0 + r * np.cos(ang[a]), v0 + r * np.sin(ang[a])))
        ok.append(True)
    pts, ok = np.array(pts), np.array(ok)
    ok &= np.abs(contrast) > 0.5 * np.median(np.abs(contrast))
    return pts[ok]


def _cone_residual_px(C, rays, radius, f):
    D = np.linalg.norm(C)
    beta = np.arccos(np.clip(rays @ (C / D), -1, 1))
    return f * (beta - np.arcsin(np.clip(radius / D, 0, 1)))


def fit_sphere_centre(edge_uv, cam, radius, iters=4):
    """Centre (camera frame) of a sphere of known radius from its silhouette edge points.

    The tangent rays make a constant angle alpha with the axis through the centre:
    rays . w = 1 with w = axis / cos(alpha) is linear in w; sin(alpha) = R / D. Then a
    robust geometric refinement with outlier rejection (a stand or a clipped edge)."""
    rays = cam.rays(edge_uv)
    w = np.linalg.lstsq(rays, np.ones(len(rays)), rcond=None)[0]
    cos_a = 1.0 / np.linalg.norm(w)
    C = (w / np.linalg.norm(w)) * radius / np.sqrt(max(1 - cos_a ** 2, 1e-12))
    f = float(cam.fx)
    inl = np.ones(len(rays), bool)
    for _ in range(iters):
        C = least_squares(lambda c: _cone_residual_px(c, rays[inl], radius, f), C,
                          loss="soft_l1", f_scale=0.1, x_scale=np.full(3, 1e-3)).x
        res = _cone_residual_px(C, rays, radius, f)
        s = 1.4826 * np.median(np.abs(res[inl] - np.median(res[inl])))
        new = np.abs(res) < max(4 * s, 0.05)
        if (new == inl).all():
            break
        inl = new
    res = _cone_residual_px(C, rays, radius, f)
    return C, {"n_edges": int(len(rays)), "inliers": int(inl.sum()),
               "rms_px": float(np.sqrt(np.mean(res[inl] ** 2))), "image_radius_px": float(f * radius / np.linalg.norm(C))}


def sphere_roi(cam, C, radius, pad_px=4):
    """Pixel-index bbox (r0, r1, c0, c1) around the sphere's image."""
    D = np.linalg.norm(C)
    uv = cam.project(C[None])[0]
    rpx = float(cam.fx) * radius / np.sqrt(max(D ** 2 - radius ** 2, 1e-12)) * 1.2 + pad_px
    c0, c1 = int(max(0, uv[0] - rpx)), int(min(cam.width, uv[0] + rpx + 1))
    r0, r1 = int(max(0, uv[1] - rpx)), int(min(cam.height, uv[1] + rpx + 1))
    return r0, r1, c0, c1


def ray_sphere(rays, C, radius):
    """First intersection distance of unit rays from the origin with the sphere (nan = miss)."""
    b = rays @ C
    disc = b ** 2 - (C @ C - radius ** 2)
    return np.where(disc >= 0, b - np.sqrt(np.clip(disc, 0, None)), np.nan)


def find_highlight(img, cam, C, radius, margin_px=3.0, win=7):
    """Sub-pixel centroid (COLMAP u, v) of the brightest spot on the ball, and its peak."""
    r0, r1, c0, c1 = sphere_roi(cam, C, radius)
    roi = img[r0:r1, c0:c1].astype(np.float64)
    jj, ii = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
    uv = np.stack([jj.ravel() + 0.5, ii.ravel() + 0.5], -1)
    rays = cam.rays(uv)
    D = np.linalg.norm(C)
    alpha = np.arcsin(radius / D)
    inside = (np.arccos(np.clip(rays @ (C / D), -1, 1)) < alpha - margin_px / float(cam.fx)).reshape(roi.shape)
    sm = cv2.GaussianBlur(roi, (0, 0), 1.0)
    k = np.argmax(np.where(inside, sm, -np.inf))
    pi, pj = np.unravel_index(k, roi.shape)
    a, b = max(0, pi - win), min(roi.shape[0], pi + win + 1)
    c, d = max(0, pj - win), min(roi.shape[1], pj + win + 1)
    W = roi[a:b, c:d]
    peak = float(W.max())
    w = np.clip(W - 0.5 * peak, 0, None)
    yy, xx = np.mgrid[a:b, c:d]
    u = (w * (xx + c0 + 0.5)).sum() / w.sum()
    v = (w * (yy + r0 + 0.5)).sum() / w.sum()
    # the spot should be unique: the next-brightest place away from it must be much dimmer
    far = inside.copy()
    far[max(0, pi - 3 * win):pi + 3 * win + 1, max(0, pj - 3 * win):pj + 3 * win + 1] = False
    second = float(sm[far].max()) if far.any() else 0.0
    return np.array([u, v]), {"peak": peak, "second_to_peak": second / max(float(sm.max()), 1e-12)}


def reflected_ray(uv, cam, C, radius):
    """Surface point S and unit reflected direction of the camera ray through uv."""
    d = cam.rays(np.asarray(uv, float).reshape(1, 2))[0]
    t = ray_sphere(d[None], C, radius)[0]
    if not np.isfinite(t):
        raise ValueError("highlight ray misses the sphere")
    S = t * d
    N = (S - C) / radius
    return S, d - 2 * (d @ N) * N


def triangulate(points, dirs):
    """Least-squares point closest to a set of rays; residual distances and a 1-sigma
    along-worst-axis uncertainty from the residual scatter."""
    points, dirs = np.asarray(points), np.asarray(dirs)
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for S, r in zip(points, dirs):
        M = np.eye(3) - np.outer(r, r)
        A += M
        b += M @ S
    P = np.linalg.solve(A, b)
    v = P - points
    t = np.einsum("ij,ij->i", v, dirs)
    dist = np.linalg.norm(v - t[:, None] * dirs, axis=1)
    dof = max(2 * len(points) - 3, 1)
    s2 = float((dist ** 2).sum() / dof)
    worst = float(np.sqrt(s2 * np.linalg.eigvalsh(np.linalg.inv(A)).max()))
    return P, {"ray_distance_mm": (dist * 1000).tolist(), "behind": bool((t <= 0).any()),
               "sigma_worst_axis_mm": worst * 1000, "n_rays": int(len(points))}
