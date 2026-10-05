"""Synthetic pilot capture: calibration targets and controls, rendered with the same
lights, forward model and noise as gwps.synth and written in the real formats
(16-bit linear PNG, calib_manifest.csv, board.json, cameras.txt).

- Balls: a mirror ball and a matte (primer, albedo 0.8) ball of the same radius on the
  same stand (a thin rod below the ball), swapped at each position. The silhouette is a
  backlit frame of the matte ball. `swap_sigma_m` offsets the mirror ball's centre from
  the matte ball's at each position (seat repeatability).
- Mirror highlights sit at the exact specular point (1-D solve in the plane of camera,
  centre and light), blurred by a 1.2 px PSF and saturated.
- Fabric board: ChArUco frame with Lambertian fabric in the middle, at several tilts and
  heights, per PS light plus ambient.
- Intrinsics: the same board seen through a distorted OPENCV camera at 15 poses.
All other captures are rendered PINHOLE, i.e. as if already undistorted.
Noise is added only near the target (the pipeline never looks elsewhere); files stay small.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import brentq

from .camera import Camera, OpenCVCamera, write_colmap_txt
from .charuco import BoardConfig
from .sphere import ray_sphere
from .synth import AMBIENT, FULL_WELL, READ_NOISE, TARGET, base_pose, studio_lights

PILOT_CAM = Camera(1, "PINHOLE", 1920, 1440, 3000.0, 3000.0, 990.3, 708.9)
DIST_PARAMS = [3000.0, 3000.6, 990.3, 708.9, -0.05, 0.012, 2e-4, -3e-4]
BALL_R = 0.040
STAND_R = 0.008
MATTE_ALBEDO = 0.8
BOARD = BoardConfig("DICT_5X5_250", 11, 9, 0.03, 0.022, False, (0.06, 0.06, 0.27, 0.21), 0.0)
SS = 3
CHART_LAYOUT = {"origin_mm": [82.5, 85.5], "pitch_mm": 33.0, "sample_mm": 18.0, "rows": 4, "cols": 6}


def volume_centre():
    R0, t0 = base_pose()
    return R0 @ TARGET + t0


def ball_positions(n=6):
    off = np.array([(-0.15, -0.22, -0.08), (0.15, -0.22, 0.08), (-0.15, 0.0, 0.08),
                    (0.15, 0.0, -0.08), (-0.15, 0.22, -0.08), (0.15, 0.22, 0.08)])
    if n == 3:
        off = off[[0, 3, 5]]
    return volume_centre() + off


def board_placements():
    """name -> (tilt about the vertical axis, tilt about the horizontal axis, height offset in m).
    'Face-on' placements are tilted 20 deg about the horizontal axis: a square-on board's normal
    rests on weak perspective and was 0.07-0.17 deg off from sub-0.05 px corner errors alone,
    against <= 0.054 deg at 20 deg (see CLAUDE.md, stage 6)."""
    return {"f+20_h0": (0, 20, 0.0), "f-20_hup": (0, -20, -0.18), "f+20_hdn": (0, 20, 0.18),
            "t+30_h0": (30, 4, 0.0), "t-30_h0": (-30, 4, 0.0), "t+50_h0": (50, 4, 0.0), "t-50_h0": (-50, 4, 0.0)}


def _Rx(d):
    a = np.radians(d)
    return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])


def _Ry(d):
    a = np.radians(d)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def board_pose(tilt_deg, tilt_x_deg, height_m, cfg=BOARD, depth=1.45):
    R = _Ry(tilt_deg) @ _Rx(tilt_x_deg)
    centre_b = np.array([cfg.size_m[0] / 2, cfg.size_m[1] / 2, 0])
    Pc = volume_centre() + np.array([0, height_m, depth - volume_centre()[2]])
    return R, Pc - R @ centre_b


# ---------------------------------------------------------------- helpers
def _subsamples(r0, r1, c0, c1, ss):
    o = (np.arange(ss) + 0.5) / ss
    u = (np.arange(c0, c1)[:, None] + o[None]).ravel()
    v = (np.arange(r0, r1)[:, None] + o[None]).ravel()
    return np.meshgrid(u, v)


def _down(A, ss):
    h, w = A.shape[0] // ss, A.shape[1] // ss
    return A.reshape((h, ss, w, ss) + A.shape[2:]).mean(axis=(1, 3))


def _pinhole_rays(cam, U, V):
    d = np.stack([(U - cam.cx) / cam.fx, (V - cam.cy) / cam.fy, np.ones_like(U)], -1).reshape(-1, 3)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _rays(cam, U, V):
    """Unit rays through (sub-)pixel positions for a pinhole or a distorted camera."""
    if isinstance(cam, OpenCVCamera):
        return cam.rays(np.stack([np.ravel(U), np.ravel(V)], -1))
    return _pinhole_rays(cam, U, V)


def _finish(full_shape, roi, img_roi, rng, noise, blur=0.5, full=None):
    out = np.zeros(full_shape, np.float64) if full is None else full.astype(np.float64)
    r0, r1, c0, c1 = roi
    x = cv2.GaussianBlur(img_roi, (0, 0), blur) if blur > 0 else img_roi
    if noise:
        x = rng.poisson(np.clip(x, 0, None) * FULL_WELL) / FULL_WELL + rng.normal(0, READ_NOISE, x.shape)
    out[r0:r1, c0:c1] = x
    if noise and full is not None:
        m = np.ones(full_shape, bool)
        m[r0:r1, c0:c1] = False
        out[m] = rng.poisson(np.clip(out[m], 0, None) * FULL_WELL) / FULL_WELL + rng.normal(0, READ_NOISE, m.sum())
    return np.clip(out, 0, 1)


def write_png16(path, img):
    """16-bit linear PNG; colour images are given as RGB and written in OpenCV's BGR order."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    x = np.round(np.clip(img, 0, 1) * 65535).astype(np.uint16)
    cv2.imwrite(str(path), x[..., ::-1] if x.ndim == 3 else x)


# ---------------------------------------------------------------- balls
def specular_point(C, R, P):
    """Point on a mirror sphere (centre C, radius R) reflecting light P into the camera at 0."""
    e1 = -C / np.linalg.norm(C)
    w = P - C
    e2 = w - (w @ e1) * e1
    e2 /= np.linalg.norm(e2)
    phiL = np.arctan2(w @ e2, w @ e1)

    def f(phi):
        N = np.cos(phi) * e1 + np.sin(phi) * e2
        S = C + R * N
        return (-S / np.linalg.norm(S)) @ N - ((P - S) / np.linalg.norm(P - S)) @ N

    phi = brentq(f, 0.0, phiL, xtol=1e-14)
    return C + R * (np.cos(phi) * e1 + np.sin(phi) * e2)


def _ball_roi(cam, C, extra_down_px=0):
    uv = cam.project(C[None])[0]
    r = cam.fx * BALL_R / C[2] * 1.15 + 12
    return (int(max(0, uv[1] - r)), int(min(cam.height, uv[1] + r + extra_down_px)),
            int(max(0, uv[0] - r)), int(min(cam.width, uv[0] + r)))


def _ball_hits(d, C):
    tb = ray_sphere(d, C, BALL_R)
    a = d[:, 0] ** 2 + d[:, 2] ** 2
    b = -2 * (d[:, 0] * C[0] + d[:, 2] * C[2])
    c = C[0] ** 2 + C[2] ** 2 - STAND_R ** 2
    disc = b * b - 4 * a * c
    tr = np.where(disc >= 0, (-b - np.sqrt(np.clip(disc, 0, None))) / (2 * a), np.nan)
    tr = np.where(tr * d[:, 1] >= C[1] + np.sqrt(BALL_R ** 2 - STAND_R ** 2), tr, np.nan)
    ball = np.isfinite(tb) & ~(np.isfinite(tr) & (tr < tb))
    rod = np.isfinite(tr) & ~ball
    return ball, rod, tb, tr


def render_ball_frames(cam, C_matte, C_mirror, lights, rng, noise=True, full_rays=None, extras=False, drift=0.0):
    """-> dict of full-size images: matte L<id>, matte ambient, silhouette, mirror L<id>, mirror ambient.
    full_rays: optional precomputed per-pixel rays (H*W, 3) for the silhouette frame.
    extras: also a two-light combination frame (additivity) and the first light repeated with
    its output scaled by (1 + drift) (end-of-session drift check)."""
    H, W = cam.height, cam.width
    roi = _ball_roi(cam, C_matte)
    U, V = _subsamples(*roi, SS)
    d = _rays(cam, U, V)
    out = {}
    for name, C in (("matte", C_matte), ("mirror", C_mirror)):
        ball, rod, tb, tr = _ball_hits(d, C)
        S = tb[:, None] * d
        n = np.zeros_like(d)
        n[ball] = (S[ball] - C) / BALL_R
        X = np.where(rod[:, None], tr[:, None] * d, S)
        n[rod] = np.stack([X[rod, 0] - C[0], np.zeros(rod.sum()), X[rod, 2] - C[2]], -1) / STAND_R
        direct = {}
        for L in lights:
            img = np.zeros(len(d))
            b, _ = L.light_vector(np.where((ball | rod)[:, None], X, 1.0))
            shade = np.clip(np.einsum("ij,ij->i", n, b), 0, None)
            img[rod] = 0.1 * (shade[rod] + AMBIENT)
            if name == "matte":
                img[ball] = MATTE_ALBEDO * (shade[ball] + AMBIENT)
                dimg = np.zeros(len(d))
                dimg[ball] = MATTE_ALBEDO * shade[ball]
                dimg[rod] = 0.1 * shade[rod]
                direct[L.id] = dimg
            else:
                sp = specular_point(C, BALL_R, L.position)
                u_s, v_s = cam.project(sp[None])[0]
                g = 4.0 * np.exp(-((U.ravel() - u_s) ** 2 + (V.ravel() - v_s) ** 2) / (2 * 1.2 ** 2))
                img[ball] = 0.005 + g[ball]
            out[f"{name}_L{L.id}"] = _finish((H, W), roi, _down(img.reshape(U.shape), SS), rng, noise)
        amb = np.zeros(len(d))
        amb[rod] = 0.1 * AMBIENT
        amb[ball] = (MATTE_ALBEDO if name == "matte" else 0.0) * AMBIENT + (0.005 if name == "mirror" else 0)
        out[f"{name}_ambient"] = _finish((H, W), roi, _down(amb.reshape(U.shape), SS), rng, noise)
        if name == "matte" and extras:
            ids = [L.id for L in lights]
            combo = direct[ids[0]] + direct[ids[1]] + amb
            out[f"matte_combo_{ids[0]}+{ids[1]}"] = _finish((H, W), roi, _down(combo.reshape(U.shape), SS), rng, noise)
            late = (1 + drift) * direct[ids[0]] + amb
            out[f"matte_drift_{ids[0]}"] = _finish((H, W), roi, _down(late.reshape(U.shape), SS), rng, noise)
    # backlit silhouette of the matte ball: bright, slightly graded backdrop; ball and rod dark
    jj, ii = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    backdrop = 0.6 + 0.2 * jj / W
    full = backdrop.copy()
    dfull = full_rays if full_rays is not None else _rays(cam, jj, ii)
    b1, r1, _, _ = _ball_hits(dfull, C_matte)
    full[(b1 | r1).reshape(H, W)] = 0.03
    ball, rod, _, _ = _ball_hits(d, C_matte)
    sil = 0.6 + 0.2 * U.ravel() / W
    sil[ball | rod] = 0.03
    out["silhouette"] = _finish((H, W), roi, _down(sil.reshape(U.shape), SS), rng, noise, full=full)
    return out


# ---------------------------------------------------------------- boards
def board_albedo_texture(cfg=BOARD, px_per_m=10000):
    Wt, Ht = int(round(cfg.size_m[0] * px_per_m)), int(round(cfg.size_m[1] * px_per_m))
    img = cfg.board().generateImage((Wt, Ht), marginSize=0, borderBits=1)
    alb = 0.04 + 0.81 * img.astype(np.float32) / 255
    if cfg.fabric_region_m is not None:
        x0, y0, x1, y1 = (np.array(cfg.fabric_region_m) * px_per_m).round().astype(int)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        xb, yb = (xx + 0.5) / px_per_m, (yy + 0.5) / px_per_m
        alb[y0:y1, x0:x1] = (0.45 + 0.08 * np.sin(2 * np.pi * xb / 0.0015) * np.sin(2 * np.pi * yb / 0.0015)
                             + 0.05 * np.sin(2 * np.pi * xb / 0.07) * np.cos(2 * np.pi * yb / 0.05))
    return alb, px_per_m


def _board_roi(cam, R, t, cfg, margin=12):
    w, h = cfg.size_m
    corners = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]]) @ R.T + t
    uv = cam.project(corners)
    c0, r0 = np.floor(uv.min(0)).astype(int) - margin
    c1, r1 = np.ceil(uv.max(0)).astype(int) + margin
    return max(0, r0), min(cam.height, r1), max(0, c0), min(cam.width, c1)


def _plane_hit(d, R, t):
    nb = R[:, 2]
    s = (nb @ t) / (d @ nb)
    X = s[:, None] * d
    Xb = (X - t) @ R
    return X, Xb


class DistortedRays:
    """Unit rays of every sub-sample of a distorted camera, computed once (float32)."""

    def __init__(self, cam, ss=2):
        self.ss = ss
        U, V = _subsamples(0, cam.height, 0, cam.width, ss)
        self.rays = cam.rays(np.stack([U.ravel(), V.ravel()], -1)).astype(np.float32).reshape(U.shape + (3,))

    def __call__(self, roi):
        r0, r1, c0, c1 = roi
        s = self.ss
        return self.rays[r0 * s:r1 * s, c0 * s:c1 * s].reshape(-1, 3).astype(np.float64)


def render_board(cam, R, t, tex, px_per_m, lights, rng, noise=True, cfg=BOARD, rays=None, uniform=None,
                 pixel_rays=None, camera_mix=None):
    """Per-light images (dict id -> image) plus 'ambient'; or a single uniformly lit image
    if uniform is a float. rays: a DistortedRays for a distorted camera (else pinhole)."""
    H, W = cam.height, cam.width
    roi = _board_roi(cam, R, t, cfg)
    ss = rays.ss if rays is not None else SS
    U, V = _subsamples(*roi, ss)
    d = rays(roi) if rays is not None else _rays(cam, U, V)
    _, Xb = _plane_hit(d, R, t)
    alb = cv2.remap(tex, (Xb[:, 0] * px_per_m - 0.5).astype(np.float32).reshape(U.shape),
                    (Xb[:, 1] * px_per_m - 0.5).astype(np.float32).reshape(U.shape),
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alb = _down(alb, ss)
    if uniform is not None:
        return _finish((H, W), roi, uniform * alb, rng, noise)
    jj, ii = np.meshgrid(np.arange(roi[2], roi[3]) + 0.5, np.arange(roi[0], roi[1]) + 0.5)
    X, _ = _plane_hit(pixel_rays(roi) if pixel_rays is not None else _rays(cam, jj, ii), R, t)
    nrm = R @ np.array([0, 0, -1.0])
    rgb = alb.ndim == 3
    full = (H, W, 3) if rgb else (H, W)
    mix = (lambda x: x @ camera_mix.T) if (rgb and camera_mix is not None) else (lambda x: x)
    out = {}
    for L in lights:
        b, _ = L.light_vector(X)
        shade = np.clip(b @ nrm, 0, None).reshape(alb.shape[:2])
        sh = shade[..., None] if rgb else shade
        out[L.id] = _finish(full, roi, mix(alb * (sh + AMBIENT)), rng, noise)
    out["ambient"] = _finish(full, roi, mix(alb * AMBIENT), rng, noise)
    return out


def intrinsics_poses(n=15, seed=3):
    rng = np.random.default_rng(seed)
    cam = OpenCVCamera(1, PILOT_CAM.width, PILOT_CAM.height, DIST_PARAMS)
    targets = [(0.5, 0.5), (0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.8), (0.5, 0.2), (0.5, 0.8),
               (0.2, 0.5), (0.8, 0.5), (0.35, 0.35), (0.65, 0.35), (0.35, 0.65), (0.65, 0.65), (0.5, 0.5), (0.3, 0.5)]
    poses = []
    for k in range(n):
        z = rng.uniform(0.75, 1.1)
        R = _Ry(rng.uniform(-35, 35)) @ _Rx(rng.uniform(-35, 35))
        u, v = targets[k % len(targets)]
        ray = cam.rays(np.array([[u * cam.width, v * cam.height]]))[0]
        centre = ray / ray[2] * z
        poses.append((R, centre - R @ np.array([BOARD.size_m[0] / 2, BOARD.size_m[1] / 2, 0])))
    return poses


# ---------------------------------------------------------------- dataset
def _chart_layout():
    from .albedo import ChartLayout
    return ChartLayout(**CHART_LAYOUT)


def chart_texture_rgb(cfg=BOARD, px_per_m=10000, layout=None, patch_mm=30.0):
    """The ChArUco board with a ColorChecker in its window (RGB, linear-sRGB albedo), and the
    reference reflectances as rendered (clipped to [0.005, 0.95]: one reference patch lies
    slightly outside sRGB)."""
    from .albedo import ChartLayout, reference_linear_srgb
    layout = layout or _chart_layout()
    grey, ppm = board_albedo_texture(cfg, px_per_m)
    tex = np.repeat(grey[..., None], 3, -1)
    x0, y0, x1, y1 = (np.array(cfg.fabric_region_m) * ppm).round().astype(int)
    tex[y0:y1, x0:x1] = 0.04                                    # the chart's black surround
    ref, _ = reference_linear_srgb()
    ref = np.clip(ref, 0.005, 0.95)
    half = patch_mm / 2000 * ppm
    for (cx, cy), val in zip(layout.patch_centres_m() * ppm, ref):
        tex[int(round(cy - half)):int(round(cy + half)), int(round(cx - half)):int(round(cx + half))] = val
    return tex.astype(np.float32), ppm, ref


def radiometry_frames(out, rng, shape=(480, 640), n=10, read=READ_NOISE, full_well=FULL_WELL,
                      exposures=(1 / 250, 1 / 125, 1 / 60, 1 / 30, 1 / 15, 1 / 8, 1 / 4, 1 / 2), shoulder=0.92):
    """Dark and flat stacks (flat = evenly lit card with a gentle illumination gradient) and an
    exposure sweep of the card whose top exposures run into a soft saturation shoulder."""
    H, W = shape
    yy, xx = np.mgrid[0:H, 0:W]
    card = 0.3 + 0.35 * (xx / W) + 0.1 * (yy / H)             # spread of levels for photon transfer

    def expose(signal):
        x = rng.poisson(np.clip(signal, 0, None) * full_well) / full_well + rng.normal(0, read, signal.shape)
        x = np.where(x > shoulder, shoulder + (1 - shoulder) * np.tanh((x - shoulder) / (1 - shoulder)), x)
        return np.clip(x, 0, 1)

    rows = []
    for k in range(n):
        for tgt, sig in (("dark", np.zeros(shape)), ("flat", card)):
            p = f"calib/{tgt}/{tgt[0]}{k}/all.png"
            write_png16(out / p, expose(sig))
            rows.append([tgt, f"{tgt[0]}{k}", 1, "all", p, ""])
    for k, t in enumerate(exposures):
        p = f"calib/sweep/e{k}/all.png"
        write_png16(out / p, expose(np.full(shape, 0.5 * t / (1 / 8))))
        rows.append(["sweep", f"e{k}", 1, "all", p, repr(t)])
    return rows


def make_pilot(out, n_positions=6, swap_sigma_m=20e-6, noise=True, seed=0,
               parts=("balls", "board", "intrinsics", "radiometry"), distorted=False, drift=0.003):
    """distorted=True renders every capture through the OPENCV camera (DIST_PARAMS), as a real
    lens would; the pipeline then needs stage-C intrinsics and undistort_lit.py before stage 6."""
    out = Path(out)
    rng = np.random.default_rng(seed)
    cam = OpenCVCamera(1, PILOT_CAM.width, PILOT_CAM.height, DIST_PARAMS) if distorted else PILOT_CAM
    rays1 = DistortedRays(cam, ss=1) if distorted else None
    rays2 = DistortedRays(cam, ss=2) if distorted else None
    lights, _ = studio_lights()
    rows = []
    truth = {"lights": {L.id: {"position_m": L.position.tolist(), "E": L.E, "axis": L.axis.tolist(), "mu": L.mu}
                        for L in lights},
             "ball_radius_m": BALL_R, "matte_centres_m": {}, "mirror_centres_m": {}, "board_poses": {},
             "distorted_camera_params": DIST_PARAMS, "swap_sigma_m": swap_sigma_m, "distorted": distorted,
             "radiometry": {"read": READ_NOISE, "full_well": FULL_WELL, "drift": drift, "saturation_shoulder": 0.92}}
    for k, Cm in enumerate(ball_positions(n_positions) if "balls" in parts else []):
        pos = f"p{k}"
        Cr = Cm + rng.normal(0, swap_sigma_m, 3)
        truth["matte_centres_m"][pos] = Cm.tolist()
        truth["mirror_centres_m"][pos] = Cr.tolist()
        fr = render_ball_frames(cam, Cm, Cr, lights, rng, noise,
                                full_rays=rays1.rays.reshape(-1, 3).astype(np.float64) if distorted else None,
                                extras=(k == 0 and "radiometry" in parts), drift=drift)
        for key, img in fr.items():
            if key == "silhouette":
                tgt, lid = "matte_ball", "silhouette"
            elif key.startswith("matte_combo_"):
                tgt, lid = "matte_ball", key[len("matte_combo_"):]
            elif key.startswith("matte_drift_"):
                p = f"calib/drift/end/{key[len('matte_drift_'):]}.png"
                write_png16(out / p, img)
                rows.append(["drift", "end", 1, key[len("matte_drift_"):], p, ""])
                continue
            else:
                kind, what = key.split("_", 1)
                tgt = f"{kind}_ball"
                lid = "ambient" if what == "ambient" else what[1:]
            p = f"calib/{tgt}/{pos}/{lid}.png"
            write_png16(out / p, img)
            rows.append([tgt, pos, 1, lid, p, ""])
    tex, ppm = board_albedo_texture() if ("board" in parts or "intrinsics" in parts) else (None, None)
    for name, (tilt, tilt_x, h) in (board_placements().items() if "board" in parts else []):
        R, t = board_pose(tilt, tilt_x, h)
        truth["board_poses"][name] = {"R": R.tolist(), "t": t.tolist()}
        for lid, img in render_board(cam, R, t, tex, ppm, lights, rng, noise, rays=rays2, pixel_rays=rays1).items():
            p = f"calib/fabric_board/{name}/{lid}.png"
            write_png16(out / p, img)
            rows.append(["fabric_board", name, 1, lid, p, ""])
    if "intrinsics" in parts:
        dcam = OpenCVCamera(1, cam.width, cam.height, DIST_PARAMS)
        drays = rays2 if distorted else DistortedRays(dcam, ss=2)
        for k, (R, t) in enumerate(intrinsics_poses()):
            truth["board_poses"][f"i{k}"] = {"R": R.tolist(), "t": t.tolist()}
            img = render_board(dcam, R, t, tex, ppm, lights, rng, noise, rays=drays, uniform=0.6)
            p = f"calib/intrinsics/i{k}/all.png"
            write_png16(out / p, img)
            rows.append(["intrinsics", f"i{k}", 1, "all", p, ""])
    if "radiometry" in parts:
        rows += radiometry_frames(out, rng)
    if "colour" in parts:
        from .synth import CAMERA_MIX
        ctex, cppm, cref = chart_texture_rgb()
        R, t = board_pose(0, 20, 0.0)
        truth["board_poses"]["c0"] = {"R": R.tolist(), "t": t.tolist()}
        truth["chart_reference_rendered"] = cref.tolist()
        truth["camera_mix"] = CAMERA_MIX.tolist()
        for lid, img in render_board(cam, R, t, ctex, cppm, lights, rng, noise, rays=rays2, pixel_rays=rays1,
                                     camera_mix=CAMERA_MIX).items():
            p = f"calib/colour_chart/c0/{lid}.png"
            write_png16(out / p, img)
            rows.append(["colour_chart", "c0", 1, lid, p, ""])
        (out / "chart.json").write_text(json.dumps(CHART_LAYOUT, indent=1))
    with open(out / "calib_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "position", "camera_id", "light_id", "path", "exposure_s"])
        w.writerows(rows)
    BOARD.save(out / "board.json")
    if distorted:
        (out / "intrinsics_true.json").write_text(json.dumps(cam.to_json(), indent=1))
    else:
        write_colmap_txt(out / "pinhole", {1: cam}, {})
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    from .lights import lights_to_json
    (out / "lights_true.json").write_text(json.dumps(lights_to_json({1: ("camera", lights)}), indent=1))
    return out
