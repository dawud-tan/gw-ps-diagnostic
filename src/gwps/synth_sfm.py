"""Synthetic garment session for the masks and the COLMAP chain (stage_masks.py, stage_sfm.py).

The torso of gwps.synth stands on a textured turntable top in front of a static textured studio
(a backdrop wall and the turntable's static base), seen through the distorted pilot camera
(calib_synth.DIST_PARAMS). Written in the formats capture.py produces:

  steps/stepNNNN/sfm.png         SfM image, all eight PS lights on, 8-bit sRGB as
                                 develop_for_colmap would write it
  steps/stepNNNN/silhouette.png  backlit frame (backdrop lit from behind, PS lights off), 16-bit linear
                                 (not with the black backdrop)
  steps/stepNNNN/ambient.png     all lights off, 16-bit linear
  manifest.csv                   the silhouette and ambient rows (PS frames are not rendered here)
  calib/metric_board/m<k>/all.png, calib_manifest.csv   ChArUco board upright on the turntable
  board.json, intrinsics_true.json
  static_exclude.png             the turntable's base, as an operator would paint it
  truth.json, truth_masks/       poses in the turntable frame (origin at the turntable top's centre,
                                 +Z up the axis, step 0's camera on -Y), camera height, axis distance,
                                 and each view's true object mask

Everything that turns (torso, turntable top, metric board) carries non-periodic 3-D value noise, so
SIFT has something to match; the studio has its own static texture, which is exactly what the masks
must keep away from COLMAP. Cast shadows are not rendered (SfM does not need them).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from .calib_synth import BOARD, DIST_PARAMS, PILOT_CAM, DistortedRays, board_albedo_texture, render_board, write_png16
from .camera import OpenCVCamera
from .raycast import Caster
from .synth import AMBIENT, BUMP_HEIGHT, BUMP_SIGMA, FULL_WELL, READ_NOISE, Rz, bump_centre, studio_lights, torso_mesh

CAM_CENTRE = np.array([0.0, -1.6, 0.30])   # turntable frame: origin at the turntable top's centre
TARGET = np.array([0.0, 0.0, 0.28])
DISC_R, DISC_H = 0.25, 0.03                # turntable top: z in [-DISC_H, 0]
BASE_R, BASE_Z0 = 0.10, -1.0               # static base below the top
BACKDROP_Z = 2.4                           # backdrop wall, z_cam (m)
METRIC_BOARD_OFFSET_Y = -0.14              # the board stands 14 cm in front of the axis


def look_at(C, T):
    f = (T - C) / np.linalg.norm(T - C)
    x = np.cross(f, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    R0 = np.stack([x, np.cross(f, x), f])
    return R0, -R0 @ C


def step_poses(angles_deg):
    """The turntable turns the garment by +angle about Z; COLMAP sees the camera orbit."""
    R0, t0 = look_at(CAM_CENTRE, TARGET)
    return [(R0 @ Rz(a), t0.copy()) for a in angles_deg]


# ---------------------------------------------------------------- texture
def _hash(i, j, k, seed):
    h = (i.astype(np.uint64) * np.uint64(73856093)) ^ (j.astype(np.uint64) * np.uint64(19349663)) \
        ^ (k.astype(np.uint64) * np.uint64(83492791)) ^ np.uint64(seed * 2654435761 % 2 ** 32)
    h = (h ^ (h >> np.uint64(13))) * np.uint64(1274126177)
    h = h ^ (h >> np.uint64(16))
    return (h & np.uint64(0xFFFFFF)).astype(np.float64) / 0xFFFFFF * 2 - 1


def value_noise(X, cell, seed):
    """Smooth lattice noise in [-1, 1] at points X (N,3), lattice spacing `cell` (m)."""
    g = X / cell
    i0 = np.floor(g).astype(np.int64)
    f = g - i0
    f = f * f * (3 - 2 * f)
    out = np.zeros(len(X))
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = ((f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1])
                     * (f[:, 2] if dz else 1 - f[:, 2]))
                out += w * _hash(i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz, seed)
    return out


def textured_albedo(X, seed, base=0.5, contrast=0.35, cells_mm=(4, 8, 16, 32)):
    amps = np.array([0.5, 0.35, 0.25, 0.15][:len(cells_mm)])
    tex = sum(a * value_noise(X, c / 1000.0, seed + k) for k, (a, c) in enumerate(zip(amps, cells_mm)))
    return np.clip(base + contrast * tex / amps.sum() * 2, 0.06, 0.95)


# ---------------------------------------------------------------- scene
def turning_mesh(bump=False):
    """Torso (garment + stand-in caps) on the turntable top; faces of the torso come first. bump:
    the 3 mm (sigma 10 mm) bump of gwps.synth, facing step 0's camera at z 0.35 m."""
    torso, garment = torso_mesh(bump=bump, z0=0.0, z1=0.60)
    disc = trimesh.creation.cylinder(radius=DISC_R, height=DISC_H, sections=180)
    disc.apply_translation([0, 0, -DISC_H / 2])
    return trimesh.util.concatenate([torso, disc]), len(torso.faces)


def static_scene(cam, rays_cam, seed=5, backdrop="lit"):
    """The studio in the camera frame (it never moves): per-pixel distance, albedo, normal (camera
    frame) and which pixels see the base. rays_cam: (H*W, 3) unit rays. backdrop 'lit': a textured
    wall (a lit sheet, shown by the PS lights in SfM frames, backlit for silhouettes); 'black': matte
    black velvet (albedo 0.015, no texture)."""
    R0, t0 = look_at(CAM_CENTRE, TARGET)
    base = trimesh.creation.cylinder(radius=BASE_R, height=-DISC_H - BASE_Z0, sections=180)
    base.apply_translation([0, 0, (BASE_Z0 - DISC_H) / 2])
    fid, loc, dist_b = Caster(base).first_hit(CAM_CENTRE, rays_cam @ R0)          # base is round: same every step
    dist_w = BACKDROP_Z / rays_cam[:, 2]
    on_base = (fid >= 0) & (dist_b < dist_w)
    dist = np.where(on_base, dist_b, dist_w)
    X = rays_cam * dist[:, None]
    n = np.tile([0, 0, -1.0], (len(X), 1))
    nb = np.zeros_like(loc)
    nb[on_base] = base.face_normals[fid[on_base]] @ R0.T
    n[on_base] = nb[on_base]
    alb = textured_albedo(X, seed, base=0.45, contrast=0.3, cells_mm=(10, 25, 60, 150))
    if backdrop == "black":
        alb = np.where(on_base, alb, 0.015)
    return {"dist": dist, "X": X, "n": n, "albedo": alb, "on_base": on_base}


def shade(X, n, albedo, lights, gain=0.125):
    """All PS lights on (no cast shadows): gain * sum_L rho (n.b_L) + ambient. gain 1/8: the
    all-lights shutter is ~3 stops shorter than the PS shutter the lights' E is scaled for."""
    s = np.zeros(len(X))
    for L in lights:
        b, _ = L.light_vector(X)
        s += np.clip(np.einsum("ij,ij->i", n, b), 0, None)
    return albedo * (gain * s + AMBIENT)


def expose(img, rng, noise=True):
    if noise:
        img = rng.poisson(np.clip(img, 0, None) * FULL_WELL) / FULL_WELL + rng.normal(0, READ_NOISE, img.shape)
    return np.clip(img, 0, 1)


def srgb8(x):
    x = np.clip(x, 0, 1)
    return np.round(255 * np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)).astype(np.uint8)


def metric_board_pose(angle_deg):
    """Board -> camera for the board upright on the turntable at `angle_deg`, facing the camera at 0.
    Board frame: origin at the top-left corner, x right, y down, z into the board (calib_synth)."""
    w, h = BOARD.size_m
    R_bw = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])     # columns: x -> +X, y -> -Z, z -> +Y
    t_bw = np.array([-w / 2, METRIC_BOARD_OFFSET_Y, 0.01 + h])
    R, t = step_poses([angle_deg])[0]
    return R @ R_bw, R @ t_bw + t


def make_session(out, n_steps=36, metric_angles=(-40, -20, 0, 20, 40), noise=True, seed=0, jitter_deg=0.0,
                 backdrop="lit", ps=False, bump=False, scale=1.0):
    """Write a synthetic garment session (see the module docstring). jitter_deg: s.d. of the
    turntable angle error per step (a hand-turned table). backdrop 'black': matte black velvet and
    no silhouette frames (masks from the SfM frames, mask_source 'sfm'). ps: also each PS light
    alone per step (16-bit linear, cast shadows, manifest light ids 1-8), for stages 3-7. bump: the
    garment carries gwps.synth's bump. scale: image size and focal length times this (1920 x 1440
    at 1; 3.125 gives 6000 x 4500, 24 MP)."""
    out = Path(out)
    rng = np.random.default_rng(seed)
    kp = [p * scale for p in DIST_PARAMS[:4]] + list(DIST_PARAMS[4:])
    cam = OpenCVCamera(1, int(round(PILOT_CAM.width * scale)), int(round(PILOT_CAM.height * scale)), kp)
    H, W = cam.height, cam.width
    rays = DistortedRays(cam, ss=1).rays.reshape(-1, 3).astype(np.float64)
    lights, _ = studio_lights()
    studio = static_scene(cam, rays, backdrop=backdrop)
    studio_sfm = shade(studio["X"], studio["n"], studio["albedo"], lights)
    backlight = 0.75 + 0.15 * (np.arange(W) + 0.5)[None].repeat(H, 0).ravel() / W    # backlit, slightly graded
    mesh, n_torso = turning_mesh(bump)
    caster = Caster(mesh)
    studio_single = {L.id: studio["albedo"] * (np.clip(np.einsum("ij,ij->i", studio["n"], L.light_vector(studio["X"])[0]),
                                                       0, None) + AMBIENT) for L in lights} if ps else {}
    angles = [360.0 * k / n_steps + (rng.normal(0, jitter_deg) if jitter_deg else 0.0) for k in range(n_steps)]
    truth = {"angles_deg": angles, "images": {}, "camera_height_m": float(CAM_CENTRE[2]),
             "axis_distance_m": float(np.hypot(*CAM_CENTRE[:2])), "intrinsics": DIST_PARAMS,
             "width": W, "height": H, "metric_angles_deg": list(map(float, metric_angles)), "metric_board_poses": {},
             "backdrop": backdrop, "ps": ps, "scale": scale, "intrinsics_scaled": kp,
             "bump_centre_world_m": bump_centre(turning_mesh(False)[0]).tolist() if bump else None,
             "bump_sigma_m": BUMP_SIGMA if bump else None, "bump_height_m": BUMP_HEIGHT if bump else None}
    rows = []
    for k, (R, t) in enumerate(step_poses(angles)):
        name = f"step{k:04d}.png"
        C = -R.T @ t
        fid, loc, dist = caster.first_hit(C, rays @ R)
        obj = (fid >= 0) & (dist < studio["dist"])
        Xc = (loc[obj] - C) @ R.T                                   # = R (loc - C), camera frame
        nc = mesh.face_normals[fid[obj]] @ R.T
        alb = textured_albedo(loc[obj], seed=11)                    # fixed to the turning objects
        sfm = studio_sfm.copy()
        sfm[obj] = shade(Xc, nc, alb, lights)
        img = srgb8(expose(sfm.reshape(H, W), rng, noise))
        p = out / "steps" / f"step{k:04d}"
        p.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p / "sfm.png"), np.repeat(img[..., None], 3, -1))
        amb = AMBIENT * studio["albedo"]
        amb[obj] = AMBIENT * alb
        write_png16(p / "ambient.png", expose(amb.reshape(H, W), rng, noise))
        rows.append([k, 1, name, "ambient", f"steps/step{k:04d}/ambient.png"])
        if ps:                                                     # each PS light alone, with cast shadows
            Rw_n = mesh.face_normals[fid[obj]]
            origin = loc[obj] + 1e-4 * Rw_n
            for L in lights:
                b, _ = L.light_vector(Xc)
                cos = np.clip(np.einsum("ij,ij->i", nc, b), 0, None)
                Pw = R.T @ (L.position - t)
                lit = (cos > 0) & (np.einsum("ij,ij->i", Rw_n, Pw - loc[obj]) > 0)
                vis = np.zeros(len(cos), bool)
                vis[lit] = ~caster.occluded(origin[lit], np.broadcast_to(Pw, origin[lit].shape))
                I = studio_single[L.id].copy()
                I[obj] = alb * (cos * vis + AMBIENT)
                write_png16(p / f"{L.id}.png", expose(I.reshape(H, W), rng, noise))
                rows.append([k, 1, name, L.id, f"steps/step{k:04d}/{L.id}.png"])
        if backdrop == "lit":
            sil = backlight.copy()
            sil[studio["on_base"]] = 0.02 * studio["albedo"][studio["on_base"]]
            sil[obj] = 0.02 * alb
            write_png16(p / "silhouette.png", expose(sil.reshape(H, W), rng, noise))
            rows.append([k, 1, name, "silhouette", f"steps/step{k:04d}/silhouette.png"])
        (out / "truth_masks").mkdir(exist_ok=True)
        cv2.imwrite(str(out / "truth_masks" / name), obj.reshape(H, W).astype(np.uint8) * 255)
        truth["images"][name] = {"R": R.tolist(), "t": t.tolist(), "centre": C.tolist(),
                                 "torso_px": int((obj & (fid < n_torso)).sum())}
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "camera_id", "colmap_image_name", "light_id", "path"])
        w.writerows(rows)
    # the operator's static exclusion: the visible base, a little grown
    base_vis = studio["on_base"].reshape(H, W) & ~obj.reshape(H, W)
    excl = cv2.dilate(base_vis.astype(np.uint8), np.ones((5, 5), np.uint8)) * 255
    cv2.imwrite(str(out / "static_exclude.png"), excl)
    # metric board frames
    tex, ppm = board_albedo_texture()
    rays2 = DistortedRays(cam, ss=2)
    crow = []
    for k, a in enumerate(metric_angles):
        Rb, tb = metric_board_pose(a)
        truth["metric_board_poses"][f"m{k}"] = {"R": Rb.tolist(), "t": tb.tolist()}
        img = render_board(cam, Rb, tb, tex, ppm, lights, rng, noise, rays=rays2, uniform=0.6)
        rel = f"calib/metric_board/m{k}/all.png"
        write_png16(out / rel, img)
        crow.append(["metric_board", f"m{k}", 1, "all", rel, ""])
    with open(out / "calib_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["target", "position", "camera_id", "light_id", "path", "exposure_s"])
        w.writerows(crow)
    BOARD.save(out / "board.json")
    (out / "intrinsics_true.json").write_text(json.dumps(cam.to_json(), indent=1))
    from .lights import lights_to_json
    (out / "lights_true.json").write_text(json.dumps(lights_to_json({1: ("camera", lights)}), indent=1))
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    mesh.export(out / "mesh_true.ply")                               # turntable frame, as stage 2 needs
    garment = np.zeros(len(mesh.faces), bool)
    garment[:n_torso] = torso_mesh(z0=0.0, z1=0.60)[1]
    np.save(out / "garment_faces_true.npy", garment)
    return out
