"""Stage 1 (principal-point gate and crop) and stage 2 (frame gates) computations.

Stage 1: GW hard-codes a centred principal point (Cx = (W-1)/2 in its integer-centred
convention, i.e. W/2 in COLMAP's), so |cx - W/2| < 0.5 px and |cy - H/2| < 0.5 px or the
undistorted SfM images and every lit image are cropped identically until it holds.
Stage 2: (a) reprojection with *our* projection (median distance close to COLMAP's mean
error, and a mean signed residual under 0.1 px per axis, which catches half-pixel shifts);
(b) mesh depth against sparse-point depth; (c) lit-image alignment against the SfM image.
"""
from __future__ import annotations

import cv2
import numpy as np


# ---------------------------------------------------------------- stage 1
def centred_crop_1d(c, n):
    """Integer crop (offset, size) of [0, n) whose centre size/2 is within 0.5 px of c - offset,
    as large as possible. Tries both parities of the size and keeps the better residual."""
    best = None
    half = int(np.floor(min(c, n - c)))
    for size in (2 * half, 2 * half - 1):
        if size <= 0:
            continue
        off = int(round(c - size / 2))
        while off + size > n:
            size -= 2
            off = int(round(c - size / 2))
        off = max(off, 0)
        res = abs((c - off) - size / 2)
        if best is None or res < best[2] - 1e-9 or (abs(res - best[2]) < 1e-9 and size > best[1]):
            best = (off, size, res)
    return best


def principal_point_gate(cam, tol=0.5):
    """-> dict(pass, dx, dy, crop=(x0, y0, W', H') or None, residual after crop)."""
    dx, dy = cam.cx - cam.width / 2, cam.cy - cam.height / 2
    if abs(dx) < tol and abs(dy) < tol:
        return {"pass": True, "dx_px": dx, "dy_px": dy, "crop": None}
    x0, w, rx = centred_crop_1d(cam.cx, cam.width)
    y0, h, ry = centred_crop_1d(cam.cy, cam.height)
    return {"pass": False, "dx_px": dx, "dy_px": dy, "crop": (x0, y0, w, h), "residual_after_crop_px": (rx, ry)}


def crop_camera(cam, crop):
    from .camera import Camera
    x0, y0, w, h = crop
    return Camera(cam.camera_id, "PINHOLE", w, h, cam.fx, cam.fy, cam.cx - x0, cam.cy - y0)


# ---------------------------------------------------------------- stage 2
def reprojection(cam, image, points):
    """Residuals of our projection of the observed 3-D points against COLMAP's POINTS2D."""
    ids = image.point3D_ids
    keep = np.array([i != -1 and i in points for i in ids], bool)
    if not keep.any():
        return None
    X = np.stack([points[i][0] for i in ids[keep]])
    err = np.array([points[i][1] for i in ids[keep]])
    Xc = image.world_to_cam(X)
    front = Xc[:, 2] > 0
    res = cam.project(Xc[front]) - image.xys[keep][front]
    d = np.linalg.norm(res, axis=1)
    return {"n": int(front.sum()), "median_px": float(np.median(d)), "mean_signed_px": res.mean(0).tolist(),
            "colmap_mean_error_px": float(np.mean(err[front])), "Xc": Xc[front], "uv": image.xys[keep][front]}


def reprojection_pass(r, tol_median=0.2, tol_signed=0.1):
    return abs(r["median_px"] - r["colmap_mean_error_px"]) <= tol_median and \
        all(abs(v) < tol_signed for v in r["mean_signed_px"])


def depth_check(caster, cam, image, Xc, uv):
    """Mesh depth along the ray through each observation versus the sparse point's depth (m)."""
    rays = cam.rays(uv) @ image.R                          # world directions
    fid, loc, _ = caster.first_hit(image.centre, rays)
    hit = fid >= 0
    if not hit.any():
        return None
    z_mesh = (loc[hit] @ image.R.T + image.t)[:, 2]
    dz = z_mesh - Xc[hit, 2]
    return {"n": int(hit.sum()), "median_abs_m": float(np.median(np.abs(dz))), "median_signed_m": float(np.median(dz))}


def _shading_normalised(x, sigma=4.0, clip=0.5):
    """Local contrast: x / local mean - 1, clipped. Removes smooth shading and gain, keeps the
    albedo texture and edges that registration needs."""
    x = np.clip(np.asarray(x, np.float32), 1e-4, None)
    return np.clip(x / cv2.GaussianBlur(x, (0, 0), sigma) - 1, -clip, clip)


def ecc_shift(ref, img):
    """Sub-pixel translation of img relative to ref (ECC on shading-normalised images), or None.
    cv2.phaseCorrelate is not used: in OpenCV 4.11 it reports 0.5 px for identical images at
    some widths (616, 617) and is biased by 0.1-0.2 px at others, too much for a 0.2 px gate."""
    if ref.shape != img.shape:
        return None
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    try:
        _, w = cv2.findTransformECC(_shading_normalised(ref), _shading_normalised(img),
                                    np.eye(2, 3, dtype=np.float32), cv2.MOTION_TRANSLATION, crit, None, 5)
    except cv2.error:
        return None
    return float(w[0, 2]), float(w[1, 2])


def linear_from_image(img):
    """8-bit images are taken as sRGB-encoded (developed SfM images); others as linear."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype == np.uint8:
        x = img.astype(np.float32) / 255
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)
    return img.astype(np.float32) / max(float(img.max()), 1e-6)


def lit_alignment(sfm_linear, lit, tol_light=0.3):
    """lit: {light_id: ambient-subtracted image}. Set level: the SfM image against the sum of
    the (unflagged) lit frames; same lighting, so the estimate is accurate to ~0.01 px: catches
    RAW crop / size / orientation mismatches of the whole set. Per light: each frame against
    the sum of the other unflagged frames, flagging the worst offender and re-measuring until
    none exceeds tol_light, so one bumped frame doesn't make its neighbours look shifted too.
    Single-light shading biases the per-light estimate by up to ~0.15 px (synthetic views), so
    single-frame bumps are caught from ~0.4 px."""
    good = set(lit)
    flagged = []
    while True:
        total = np.sum([lit[k] for k in good], axis=0)
        per = {k: ecc_shift(total - lit[k] if k in good else total, lit[k]) for k in lit}
        mag = {k: (np.inf if per[k] is None else float(np.hypot(*per[k]))) for k in good}
        bad = [k for k in good if mag[k] >= tol_light]
        if not bad or len(good) <= 2:
            break
        worst = max(bad, key=lambda k: mag[k])
        good.discard(worst)
        flagged.append(worst)
    return {"set": ecc_shift(sfm_linear, total), "per_light": per,
            "flagged": sorted(flagged + [k for k in good if mag[k] >= tol_light], key=str)}


def bracket_alignment(base_raw, bracket_raw, bracket_sat, k, tol_set=0.2, tol_light=0.3):
    """Exposure brackets: each bracket frame (raw, ambient included) against the base frame of
    the same light scaled by the exposure ratio k and clipped where the bracket clips, so both
    show the same shading and the same flat clipped areas; the expected shift is exactly 0.
    Also the sums over lights (the bracket set as a whole). Rejected first: lit_alignment's
    each-against-the-others, which a long frame's clipped regions starve of texture (a bright
    synthetic torso at 4x flagged five lights for one bumped frame), and an ECC mask over the
    clipped regions, which was unstable (2.5 px on clean frames). Measured on the synthetic
    torso at 4x, test noise: clean frames 0.03-0.06 px (bright, a quarter clipped) and up to
    0.11 px (x0.15 albedo, where the base frame's noise limits it); a frame bumped by 1 px, 0.91 px."""
    per, refs, imgs = {}, [], []
    for lid, img in bracket_raw.items():
        sat = bracket_sat[lid]
        clip = float(np.median(img[sat])) if sat.any() else 1.0
        ref = np.minimum(base_raw[lid] * np.float32(k), np.float32(clip))
        per[lid] = ecc_shift(ref, img)
        refs.append(ref)
        imgs.append(img)
    st = ecc_shift(np.sum(refs, axis=0), np.sum(imgs, axis=0))
    bad = [lid for lid, x in per.items() if x is None or np.hypot(*x) >= tol_light]
    return {"set": st, "set_pass": st is not None and float(np.hypot(*st)) < tol_set,
            "per_light": per, "flagged": sorted(bad, key=str)}


def silhouette_overlay(img8, hit, colour=(0, 0, 255), mask=None, mask_colour=(0, 200, 0)):
    """BGR image with the mesh silhouette (boundary of the hit mask, red) drawn on it, and the
    object mask's boundary (green) if given."""
    base = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR) if img8.ndim == 2 else img8[..., :3].copy()
    k = np.ones((3, 3), np.uint8)
    if mask is not None:
        base[cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0] = mask_colour
    base[cv2.morphologyEx(hit.astype(np.uint8), cv2.MORPH_GRADIENT, k) > 0] = colour
    return base
