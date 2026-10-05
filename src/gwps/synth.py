"""Synthetic turntable scene with known poses and mesh, rendered with the CLAUDE.md
forward model, written out in exactly the formats the real pipeline reads.

World: metres, origin at the turntable centre, +Z along the turntable axis. The garment
stand-in is an elliptic torso with smooth folds and panel undulations; its end caps
stand in for mannequin/turntable and are excluded via garment_faces.npy. One fixed
PINHOLE camera (off-centre principal point) and 8 LEDs fixed in the studio, so the
lights are stored in the camera frame and are the same for every turntable step.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .camera import Camera, Image, R_to_qvec, write_colmap_txt
from .lights import Light, lights_to_json
from .raycast import Caster, cast_view

W, H = 640, 480
FX = FY = 1000.0
CX, CY = 330.7, 236.2            # deliberately off-centre
CAM_CENTRE = np.array([0.0, -1.5, 0.38])
TARGET = np.array([0.0, 0.0, 0.35])
STEP_DEG = 30.0
N_STEPS = 12
READ_NOISE = 0.005               # 0.5 % of full scale
FULL_WELL = 10000.0              # electrons at full scale, for Poisson shot noise
AMBIENT = 0.03
BUMP_CENTRE_THETA = -np.pi / 2   # faces the camera at step 0
BUMP_Z = 0.35
BUMP_SIGMA = 0.010
BUMP_HEIGHT = 0.003


def Rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def base_pose():
    f = TARGET - CAM_CENTRE
    f /= np.linalg.norm(f)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(f, x)
    R0 = np.stack([x, y, f])
    return R0, -R0 @ CAM_CENTRE


def camera(scale=1.0):
    """The synthetic PINHOLE camera; scale multiplies resolution and focal length (same field of view)."""
    return Camera(1, "PINHOLE", int(round(W * scale)), int(round(H * scale)), FX * scale, FY * scale, CX * scale, CY * scale)


def poses(angles_deg):
    """Turntable rotates the garment by +angle about Z; COLMAP sees the camera orbit."""
    R0, t0 = base_pose()
    return [(R0 @ Rz(a), t0.copy()) for a in angles_deg]


def albedo(X):
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    a = (0.62 + 0.18 * np.sin(2 * np.pi * x / 0.07) * np.sin(2 * np.pi * y / 0.05 + 1) * np.sin(2 * np.pi * z / 0.09)
         + 0.10 * np.sin(2 * np.pi * (x + y + z) / 0.013))
    return np.clip(a, 0.3, 0.95)


# A synthetic camera's spectral response: camera RGB = CAMERA_MIX @ linear-sRGB radiance
# (channel mixing times per-channel gains), what a colour correction must undo.
CAMERA_MIX = np.diag([0.72, 1.0, 0.61]) @ np.array([[0.80, 0.15, 0.05], [0.10, 0.80, 0.10], [0.02, 0.18, 0.80]])


def albedo_rgb(X):
    """Linear-sRGB albedo: the grey texture tinted by smooth colour panels (red low, green in
    the middle, blue high), so a colour correction has saturated colours to get right."""
    base = albedo(X)
    z = X[:, 2]
    w = np.stack([np.clip(1 - np.abs(z - c) / 0.16, 0, 1) for c in (0.17, 0.35, 0.53)], -1)
    tints = np.array([[1.0, 0.42, 0.35], [0.38, 0.85, 0.42], [0.33, 0.45, 0.95]])
    tint = (w @ tints + (1 - w.sum(-1, keepdims=True)).clip(0) * 0.8) / np.maximum(w.sum(-1, keepdims=True), 1)
    return np.clip(base[:, None] * tint, 0.02, 0.95)


def studio_lights():
    """8 bare LEDs about 1.2-1.5 m from the garment, 30-40 deg off the camera axis, fixed
    to the studio -> camera frame. mu = 1 falloff about an axis aimed near the garment."""
    R0, t0 = base_pose()
    centre = R0 @ TARGET + t0
    rng = np.random.default_rng(7)
    lights = []
    gain = 1.45
    for j in range(8):
        az = np.radians(10 + 45 * j)
        off = np.radians(30 if j % 2 == 0 else 40)
        d = 1.2 + 0.3 * ((j * 3) % 8) / 7
        dirn = np.array([np.sin(off) * np.cos(az), np.sin(off) * np.sin(az), -np.cos(off)])
        P = centre + d * dirn
        aim = centre + rng.normal(0, 0.05, 3) - P
        lights.append(Light(j + 1, "ps", "point", P, gain * (1 + 0.1 * rng.uniform(-1, 1)),
                            aim / np.linalg.norm(aim), 1.0))
    ref = Light(99, "reference", "point", centre + np.array([0, -0.3, -1.0]), 1.0, None, 0.0)
    return lights, ref


def torso_mesh(bump=False, n_theta=340, n_z=200, z0=0.05, z1=0.65):
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    z = np.linspace(z0, z1, n_z)
    T, Z = np.meshgrid(th, z)                  # (n_z, n_theta)
    a, b = 0.18, 0.13
    r = a * b / np.sqrt((b * np.cos(T)) ** 2 + (a * np.sin(T)) ** 2)
    r = r + (0.006 * np.sin(3 * T + 2 * np.pi * Z / 0.35)
             + 0.004 * np.sin(7 * T + 1) * (0.5 + 0.5 * np.cos(2 * np.pi * Z / 0.6))
             + 0.003 * np.sin(2 * np.pi * Z / 0.12 + 2 * T))
    V = np.stack([r * np.cos(T), r * np.sin(T), Z], -1).reshape(-1, 3)
    idx = np.arange(n_z * n_theta).reshape(n_z, n_theta)
    a00, a01 = idx[:-1], np.roll(idx, -1, axis=1)[:-1]
    a10, a11 = idx[1:], np.roll(idx, -1, axis=1)[1:]
    side = np.concatenate([np.stack([a00, a01, a11], -1).reshape(-1, 3),
                           np.stack([a00, a11, a10], -1).reshape(-1, 3)])
    cb, ct = len(V), len(V) + 1
    V = np.vstack([V, [0, 0, z0], [0, 0, z1]])
    bot = np.stack([np.full(n_theta, cb), np.roll(idx[0], -1), idx[0]], -1)
    top = np.stack([np.full(n_theta, ct), idx[-1], np.roll(idx[-1], -1)], -1)
    F = np.concatenate([side, bot, top])
    garment = np.zeros(len(F), bool)
    garment[:len(side)] = True
    mesh = trimesh.Trimesh(V, F, process=False)
    if bump:
        c = bump_centre(mesh)
        d2 = ((mesh.vertices - c) ** 2).sum(1)
        h = BUMP_HEIGHT * np.exp(-d2 / (2 * BUMP_SIGMA ** 2))
        h[-2:] = 0
        mesh = trimesh.Trimesh(mesh.vertices + h[:, None] * mesh.vertex_normals, F, process=False)
    return mesh, garment


def bump_centre(mesh=None):
    if mesh is None:
        mesh, _ = torso_mesh()
    d = np.array([np.cos(BUMP_CENTRE_THETA), np.sin(BUMP_CENTRE_THETA), 0.0])
    fid, loc, _ = Caster(mesh).first_hit(np.array([[0, 0, BUMP_Z]]) + 1.0 * d, -d[None])
    return loc[0]


def board_mesh(size=0.3, n=120):
    """Flat board in the plane y = 0 facing -Y (the camera at step 0), centred on the axis."""
    g = np.linspace(-size / 2, size / 2, n)
    X, Zg = np.meshgrid(g, g + TARGET[2])
    V = np.stack([X, np.zeros_like(X), Zg], -1).reshape(-1, 3)
    idx = np.arange(n * n).reshape(n, n)
    a00, a01, a10, a11 = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    F = np.concatenate([np.stack([a00, a10, a11], -1).reshape(-1, 3),
                        np.stack([a00, a11, a01], -1).reshape(-1, 3)])
    mesh = trimesh.Trimesh(V, F, process=False)
    if mesh.face_normals[:, 1].mean() > 0:
        mesh = trimesh.Trimesh(V, F[:, ::-1], process=False)
    return mesh, np.ones(len(F), bool)


# ---------------------------------------------------------------- camera-frame normal perturbations
def _tilt(n, t_dir, alpha_rad):
    t = t_dir - np.sum(t_dir * n, -1, keepdims=True) * n
    t /= np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-12)
    out = n + np.tan(alpha_rad)[..., None] * t
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def perturb_ripple(n, u, v):
    """Fine texture: 3 px period, +-10 deg, tilting toward camera x. Sampled at pixel
    centres the phases give +10, -5, -5 deg, mean 0."""
    alpha = np.radians(10.0) * np.sin(2 * np.pi * u / 3.0 + np.pi / 6)
    return _tilt(n, np.broadcast_to([1.0, 0, 0], n.shape), alpha)


def perturb_bias(n, u, v):
    """Smooth PS-like bias fixed to the camera/lights: 4 deg tilt whose direction turns
    with image position (wavelength ~430 px across, ~960 px down)."""
    psi = 2 * np.pi * (1.5 * u / W + 0.5 * v / H)
    t = np.stack([np.cos(psi), np.sin(psi), np.zeros_like(psi)], -1)
    return _tilt(n, t, np.full(u.shape, np.radians(4.0)))


def perturb_vbias(n, u, v):
    """Adversarial bias: tilt toward camera y (world vertical) that depends only on image
    row. Turntable rotation never moves a face in v, so this looks garment-fixed."""
    alpha = np.radians(4.0) * np.sin(2 * np.pi * v / H)
    return _tilt(n, np.broadcast_to([0, 1.0, 0], n.shape), alpha)


def perturb_vbias_edge(n, u, v):
    """The blind spot where a mid-height board cannot see it: a 4 deg vertical tilt that is zero
    within 110 px of the image's middle row and ramps to full over the next 40 px, up at the
    top and down at the bottom. The mid-height synthetic board spans about +-100 px."""
    dv = v - H / 2
    alpha = np.radians(4.0) * np.sign(dv) * np.clip((np.abs(dv) - 110.0) / 40.0, 0, 1)
    return _tilt(n, np.broadcast_to([0, 1.0, 0], n.shape), alpha)


def ridges_world(nw, Xw, amp_deg=8.0, period=0.012):
    """World-space fabric detail (a corduroy-like cord): normals tilt along the surface's
    vertical tangent by amp * sin(2 pi z / period). Fixed to the garment, unlike ripple/bias."""
    alpha = np.radians(amp_deg) * np.sin(2 * np.pi * Xw[:, 2] / period)
    return _tilt(nw, np.broadcast_to([0, 0, 1.0], nw.shape), alpha)


PERTURB = {None: None, "ripple": perturb_ripple, "bias": perturb_bias, "vbias": perturb_vbias,
           "vbias_edge": perturb_vbias_edge}
WORLD_PERTURB = {"ridges": ridges_world}


# ---------------------------------------------------------------- rendering
def sheen_brdf(n, l, v, ks=0.06, p=24.0):
    """Mild fabric-like sheen: Lambertian plus a Blinn-Phong lobe, as a factor on the
    Lambertian term's (n.l): I = rho E (n.l) + ks E (n.h)^p (n.l). Not what PS assumes."""
    h = l + v
    h /= np.linalg.norm(h, axis=1, keepdims=True)
    return ks * np.clip(np.einsum("ij,ij->i", n, h), 0, None) ** p


def render_view(caster, cam, R, t, lights, albedo_fn=albedo, perturb=None, noise=False, rng=None,
                brdf=None, noise_params=None, rgb=False, world_perturb=None, exposure=1.0):
    """noise: False, or True (READ_NOISE, FULL_WELL); noise_params=(read, full_well) overrides.
    brdf: optional extra lobe (n, l, v) -> factor added to rho (see sheen_brdf; white, all channels).
    rgb: albedo_fn returns linear-sRGB albedo (N,3); images are camera RGB (CAMERA_MIX applied),
    (H,W,3), and the ground truth includes albedo_rgb.
    exposure: relative exposure time; scales every frame (lit and ambient) before noise and
    clipping, as a longer shutter does on a linear sensor."""
    read, fw = noise_params or (READ_NOISE, FULL_WELL)
    maps = cast_view(caster, cam, R, t)
    hit = maps["hit"].ravel()
    X = maps["pos_cam"].reshape(-1, 3)[hit].astype(np.float64)
    n_geo = maps["normal_cam"].reshape(-1, 3)[hit].astype(np.float64)
    uv = cam.pixel_centres()[hit]
    Xw = (X - t) @ R                       # R^T (X - t)
    n = n_geo
    if world_perturb is not None:
        n = world_perturb(n @ R, Xw) @ R.T  # perturb in the world frame, back to camera
    n = perturb(n, uv[:, 0], uv[:, 1]) if perturb else n
    rho = albedo_fn(Xw)
    rho = rho if rgb else rho[:, None]
    C = rho.shape[1]
    origin = Xw + 1e-4 * (n_geo @ R)
    npx = cam.width * cam.height
    imgs, lit_all = [], np.ones(len(X), bool)
    for L in lights:
        b, _ = L.light_vector(X)
        cos = np.einsum("ij,ij->i", n, b)
        extra = np.zeros(len(X))
        if brdf is not None:
            bl = np.linalg.norm(b, axis=1)
            extra = brdf(n, b / bl[:, None], -X / np.linalg.norm(X, axis=1, keepdims=True))
        Pw = R.T @ (L.position - t)
        vis = ~caster.occluded(origin, np.broadcast_to(Pw, origin.shape))
        vis &= np.einsum("ij,ij->i", n_geo, b) > 0
        lit_all &= vis & (cos > 0)
        I = np.zeros((npx, C))
        I[hit] = (rho + extra[:, None]) * (np.clip(cos, 0, None) * vis)[:, None] + AMBIENT * rho
        imgs.append(I)
    amb = np.zeros((npx, C))
    amb[hit] = AMBIENT * rho
    imgs.append(amb)
    out = []
    for I in imgs:
        if exposure != 1.0:
            I = I * exposure
        if rgb:
            I = I @ CAMERA_MIX.T
        if noise:
            I = rng.poisson(np.clip(I, 0, None) * fw) / fw + rng.normal(0, read, I.shape)
        I = np.clip(I, 0, 1).reshape(cam.height, cam.width, C).astype(np.float32)
        out.append(I if rgb else I[..., 0])
    gt_n = np.zeros((npx, 3), np.float32)
    gt_n[hit] = n
    unshadowed = np.zeros(npx, bool)
    unshadowed[hit] = lit_all
    gt = {"n_gt_cam": gt_n.reshape(cam.height, cam.width, 3), "hit": maps["hit"],
          "unshadowed": unshadowed.reshape(cam.height, cam.width), "face_id": maps["face_id"]}
    if rgb:
        a = np.zeros((npx, 3), np.float32)
        a[hit] = rho
        gt["albedo_rgb"] = a.reshape(cam.height, cam.width, 3)
    return out[:-1], out[-1], gt


def _sparse_points(mesh, garment, cam, pose_list, caster, n_pts=3000, seed=0):
    rng = np.random.default_rng(seed)
    fids = rng.choice(np.flatnonzero(garment), n_pts, replace=False)
    Xw = mesh.triangles_center[fids]
    pts = {i + 1: Xw[i] for i in range(n_pts)}
    obs = []
    for R, t in pose_list:
        Xc = Xw @ R.T + t
        uv = cam.project(Xc)
        inside = (uv[:, 0] > 0) & (uv[:, 0] < cam.width) & (uv[:, 1] > 0) & (uv[:, 1] < cam.height)
        C = -R.T @ t
        d = Xw - C
        dist = np.linalg.norm(d, axis=1)
        _, _, hd = caster.first_hit(C, d / dist[:, None])
        vis = inside & (np.abs(hd - dist) < 1e-4)
        obs.append((uv[vis], np.flatnonzero(vis) + 1))
    return pts, obs


def make_dataset(out, kind="torso", variant=None, noise=False, seed=0, angles=None, brdf=None,
                 noise_params=None, resolution_scale=1.0, rgb=False, tessellation=None,
                 albedo_scale=1.0, exposures=(1.0,), cam=None, heights=None):
    """Write a complete synthetic dataset. kind: torso | board. variant: None | bump |
    ripple | bias | vbias. For 'bump' the ground truth has the bump and mesh.ply lacks it.
    albedo_scale multiplies the albedo texture (a darker fabric of the same weave).
    exposures: relative exposures of the lit and ambient frames, the first being the base; more
    than one writes an exposure bracket (manifest column exposure_s = the factor, in seconds of a
    1 s base). cam overrides the PINHOLE camera (e.g. a crop at a real camera's pixel density).
    heights: per view, metres the object sits above its usual place relative to camera and
    lights (the board at another height in the image; the rig moves, the lights stay in the
    camera frame)."""
    out = Path(out)
    (out / "lit").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)
    cam = cam if cam is not None else camera(resolution_scale)
    alb = albedo_rgb if rgb else albedo
    alb_fn = alb if albedo_scale == 1.0 else (lambda X: albedo_scale * alb(X))
    exposures = [float(e) for e in exposures]
    if exposures[0] != 1.0 or any(e <= 1.0 for e in exposures[1:]):
        raise ValueError(f"exposures {exposures}: the first is the base (1.0), brackets are longer")
    bracketed = len(exposures) > 1
    if kind == "torso":
        tess = {} if tessellation is None else {"n_theta": tessellation[0], "n_z": tessellation[1]}
        mesh_gt, garment = torso_mesh(bump=(variant == "bump"), **tess)     # tessellation: (n_theta, n_z),
        mesh_test, _ = torso_mesh(bump=False, **tess)                       # e.g. GW-like face counts
        angles = angles if angles is not None else [STEP_DEG * k for k in range(N_STEPS)]
    elif kind == "board":
        mesh_gt, garment = board_mesh()
        mesh_test = mesh_gt
        angles = angles if angles is not None else [0, 30, -30, 50, -50]
    else:
        raise ValueError(kind)
    pose_list = poses(angles)
    if heights is not None:                             # object up by dz = camera and lights down by dz
        if len(heights) != len(pose_list):
            raise ValueError("heights needs one value per view")
        pose_list = [(R, t + dz * R[:, 2]) for (R, t), dz in zip(pose_list, heights)]
    lights, ref = studio_lights()
    caster = Caster(mesh_gt)
    perturb = PERTURB[variant if variant in PERTURB else None]
    images, rows = {}, []
    for k, (R, t) in enumerate(pose_list):
        name = f"step{k:04d}.png"
        stem = Path(name).stem
        kw = dict(albedo_fn=alb_fn, perturb=perturb, noise=noise, rng=rng, world_perturb=WORLD_PERTURB.get(variant),
                  brdf=sheen_brdf if brdf == "sheen" else brdf, noise_params=noise_params, rgb=rgb)
        lit, amb, gt = render_view(caster, cam, R, t, lights, **kw)
        e0 = [1.0] if bracketed else []
        for L, I in zip(lights, lit):
            np.save(out / "lit" / f"{stem}_L{L.id}.npy", I)
            rows.append([k, 1, name, L.id, f"lit/{stem}_L{L.id}.npy"] + e0)
        np.save(out / "lit" / f"{stem}_amb.npy", amb)
        rows.append([k, 1, name, "ambient", f"lit/{stem}_amb.npy"] + e0)
        for j, e in enumerate(exposures[1:], 1):        # the same view at a longer exposure
            lit_e, amb_e, _ = render_view(caster, cam, R, t, lights, exposure=e, **kw)
            for L, I in zip(lights, lit_e):
                np.save(out / "lit" / f"{stem}_L{L.id}_b{j}.npy", I)
                rows.append([k, 1, name, L.id, f"lit/{stem}_L{L.id}_b{j}.npy", e])
            np.save(out / "lit" / f"{stem}_amb_b{j}.npy", amb_e)
            rows.append([k, 1, name, "ambient", f"lit/{stem}_amb_b{j}.npy", e])
        np.savez_compressed(out / "gt" / f"{stem}.npz", **gt)
        sfm = np.clip(0.5 * np.sum([I - amb for I in lit], axis=0) + amb, 0, 1)   # the SfM/GW image: all lights on
        (out / "images").mkdir(exist_ok=True)
        sfm8 = np.round(255 * np.where(sfm <= 0.0031308, 12.92 * sfm, 1.055 * sfm ** (1 / 2.4) - 0.055)).astype(np.uint8)
        cv2.imwrite(str(out / "images" / name), sfm8[..., ::-1] if sfm8.ndim == 3 else sfm8)
        images[k + 1] = Image(k + 1, R_to_qvec(R), t, 1, name)
    pts, obs = _sparse_points(mesh_gt, garment, cam, pose_list, caster, n_pts=min(3000, int(garment.sum())))
    for (uv, pid), im in zip(obs, images.values()):
        im.xys, im.point3D_ids = uv, pid
    write_colmap_txt(out / "sparse" / "0", {1: cam}, images, pts)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "camera_id", "colmap_image_name", "light_id", "path"] + (["exposure_s"] if bracketed else []))
        w.writerows(rows)
    (out / "lights.json").write_text(json.dumps(lights_to_json({1: ("camera", lights + [ref])}), indent=1))
    mesh_test.export(out / "mesh.ply")
    mesh_gt.export(out / "mesh_gt.ply")
    np.save(out / "garment_faces.npy", garment)
    info = {"kind": kind, "variant": variant, "noise": noise, "angles_deg": list(map(float, angles)),
            "brdf": brdf if isinstance(brdf, (str, type(None))) else "custom",
            "noise_params": list(noise_params) if noise_params else None, "resolution_scale": resolution_scale,
            "rgb": rgb, "camera_mix": CAMERA_MIX.tolist() if rgb else None, "tessellation": tessellation,
            "albedo_scale": albedo_scale, "exposures": exposures, "heights_m": list(heights) if heights is not None else None,
            "camera": {"width": cam.width, "height": cam.height, "fx": cam.fx, "fy": cam.fy, "cx": cam.cx, "cy": cam.cy},
            "read_noise": READ_NOISE, "full_well": FULL_WELL, "ambient": AMBIENT}
    if variant == "bump":
        info["bump_centre_world_m"] = bump_centre().tolist()
        info["bump_sigma_m"], info["bump_height_m"] = BUMP_SIGMA, BUMP_HEIGHT
    (out / "scene.json").write_text(json.dumps(info, indent=1))
    return out
