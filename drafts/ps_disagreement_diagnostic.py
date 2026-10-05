#!/usr/bin/env python3
"""
ps_disagreement_diagnostic.py — frequency-separated check of where
Gaussian Wrapping's mesh normals disagree with photometric-stereo
normals (Research report, Recommendation #1).

Decision rule: disagreement only at sub-face scale -> bake a normal
map, stop. Disagreement at per-face (resolvable) scale, spatially
clustered -> the mesh's actual shape is wrong there; only L_PS
training-time supervision fixes that, not baking.

Assumes: Gaussian Wrapping mesh already extracted; COLMAP sparse
model with camera poses (post image_undistorter, so PINHOLE camera
model — 4 params, no distortion); per-step multi-light captures at
images_root/step_XXXX/light_NN.jpg; light_directions.json — N
calibrated unit vectors in CAMERA space, one-time rig calibration,
reused across every step.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import trimesh
import pycolmap
from PIL import Image


def solve_photometric_stereo(images: np.ndarray, light_dirs: np.ndarray,
                              shadow_floor: float = 0.02):
    """images: (N,H,W) grayscale in [0,1], N>=4. Returns camera-space
    normals (H,W,3), albedo (H,W), residual (H,W) as a confidence proxy —
    high residual flags non-Lambertian pixels (sheen, subsurface
    scattering), the report's flagged risk on fabric."""
    N, H, W = images.shape
    I = images.reshape(N, -1)
    valid = I > shadow_floor

    g = np.zeros((3, H * W))
    residual = np.zeros(H * W)
    patterns, inverse = np.unique(valid.T, axis=0, return_inverse=True)
    for pat_idx, pattern in enumerate(patterns):
        if pattern.sum() < 3:
            continue  # under-determined at this pixel; leave albedo 0
        cols = np.where(inverse == pat_idx)[0]
        L_sub = light_dirs[pattern]
        I_sub = I[pattern][:, cols]
        g[:, cols] = np.linalg.pinv(L_sub) @ I_sub
        residual[cols] = np.mean((L_sub @ g[:, cols] - I_sub) ** 2, axis=0)

    albedo = np.linalg.norm(g, axis=0)
    normals = np.divide(g, albedo, out=np.zeros_like(g), where=albedo > 1e-6)
    return (normals.T.reshape(H, W, 3), albedo.reshape(H, W),
            residual.reshape(H, W))


def load_camera_poses(colmap_sparse_path: str):
    """{image_name: (R_world_from_cam(3,3), C_world(3,), fx, fy, cx, cy)}.
    NOTE: cam_from_world.rotation / .translation sub-attribute names are
    my best reading of the pycolmap Rigid3d API, not directly confirmed
    this session — check `help(pycolmap.Rigid3d)` against your installed
    version before trusting this function blind."""
    rec = pycolmap.Reconstruction(colmap_sparse_path)
    poses = {}
    for image in rec.images.values():
        T = image.cam_from_world
        R_world_from_cam = T.rotation.matrix().T
        C_world = T.inverse().translation
        cam = rec.cameras[image.camera_id]
        fx, fy, cx, cy = cam.params[:4]  # PINHOLE, valid post-undistortion
        poses[image.name] = (R_world_from_cam, C_world, fx, fy, cx, cy)
    return poses


def project_and_collect(mesh, normals_world, confidence,
                         R_world_from_cam, C_world, fx, fy, cx, cy,
                         stride: int = 4):
    H, W, _ = normals_world.shape
    ys, xs = np.mgrid[0:H:stride, 0:W:stride]
    ys, xs = ys.ravel().astype(np.float64), xs.ravel().astype(np.float64)

    # Simple pinhole unprojection — valid because these are undistorted
    # (post image_undistorter) images, not raw distorted ones.
    dirs_cam = np.stack([(xs - cx) / fx, (ys - cy) / fy,
                          np.ones_like(xs)], axis=1)
    dirs_cam /= np.linalg.norm(dirs_cam, axis=1, keepdims=True)
    dirs_world = dirs_cam @ R_world_from_cam.T
    origins = np.repeat(C_world[None, :], len(xs), axis=0)

    locations, ray_idx, tri_idx = mesh.ray.intersects_location(
        origins, dirs_world, multiple_hits=False)

    samples: dict[int, list] = {}
    ys_i, xs_i = ys.astype(int), xs.astype(int)
    for r, t in zip(ray_idx, tri_idx):
        y, x = ys_i[r], xs_i[r]
        samples.setdefault(t, []).append(
            (normals_world[y, x], confidence[y, x]))
    return samples


def frequency_separate(mesh, all_samples: dict):
    n_faces = len(mesh.faces)
    resolvable = np.full(n_faces, np.nan)
    subface = np.full(n_faces, np.nan)
    count = np.zeros(n_faces, dtype=int)

    for f, pts in all_samples.items():
        if len(pts) < 3:
            continue
        normals = np.array([p[0] for p in pts])
        w = np.array([p[1] for p in pts]); w /= (w.sum() + 1e-9)

        mean_n = (normals * w[:, None]).sum(axis=0)
        mean_n /= (np.linalg.norm(mean_n) + 1e-9)

        resolvable[f] = np.degrees(np.arccos(
            np.clip(mesh.face_normals[f] @ mean_n, -1, 1)))
        ang = np.degrees(np.arccos(np.clip(normals @ mean_n, -1, 1)))
        subface[f] = np.sqrt((w * ang ** 2).sum())
        count[f] = len(pts)

    valid = count >= 3
    return {
        "resolvable_deg": resolvable, "subface_deg": subface,
        "count": count, "coverage": float(valid.mean()),
        "median_resolvable": float(np.nanmedian(resolvable[valid])),
        "p90_resolvable": float(np.nanpercentile(resolvable[valid], 90)),
        "median_subface": float(np.nanmedian(subface[valid])),
    }


def verdict(result: dict, resolvable_threshold_deg: float = 8.0,
            cluster_face_fraction: float = 0.05):
    r = result["resolvable_deg"]; valid = ~np.isnan(r)
    bad = float((r[valid] > resolvable_threshold_deg).mean())
    if bad < cluster_face_fraction:
        return ("BAKE", f"{bad:.1%} of faces exceed "
                f"{resolvable_threshold_deg}° — high-frequency only. "
                f"Bake a normal map, skip L_PS.")
    return ("RETRAIN_CANDIDATE", f"{bad:.1%} of faces exceed "
            f"{resolvable_threshold_deg}° — inspect the exported PLY "
            f"for spatial clustering before committing to L_PS.")


def export_diagnostic_ply(mesh, result: dict, out_path: str):
    r = result["resolvable_deg"]; valid = ~np.isnan(r)
    norm = np.clip(np.nan_to_num(r) / 20.0, 0, 1)
    colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)
    colors[:, 0] = (norm * 255).astype(np.uint8)
    colors[:, 1] = ((1 - norm) * 180).astype(np.uint8)
    colors[:, 3] = 255
    colors[~valid] = [80, 80, 80, 255]  # gray = no PS coverage there
    mesh.visual.face_colors = colors
    mesh.export(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--colmap-sparse", required=True)
    ap.add_argument("--images-root", required=True)
    ap.add_argument("--light-directions", required=True)
    ap.add_argument("--out-ply", default="diagnostic_disagreement.ply")
    ap.add_argument("--stride", type=int, default=4)
    args = ap.parse_args()

    mesh = trimesh.load(args.mesh, process=False)
    poses = load_camera_poses(args.colmap_sparse)
    light_dirs = np.array(json.load(open(args.light_directions)))

    all_samples: dict[int, list] = {}
    for step_dir in sorted(Path(args.images_root).glob("step_*")):
        image_name = f"{step_dir.name}.jpg"  # must match the COLMAP image name
        if image_name not in poses:
            continue
        R_world_from_cam, C_world, fx, fy, cx, cy = poses[image_name]

        stack = np.stack([np.asarray(Image.open(p).convert("L"), dtype=np.float64) / 255.0
                           for p in sorted(step_dir.glob("light_*.jpg"))])
        normals_cam, _, residual = solve_photometric_stereo(stack, light_dirs)
        confidence = 1.0 / (1.0 + 50.0 * residual)
        normals_world = (normals_cam.reshape(-1, 3) @ R_world_from_cam.T).reshape(normals_cam.shape)

        for face_id, pts in project_and_collect(
                mesh, normals_world, confidence, R_world_from_cam,
                C_world, fx, fy, cx, cy, stride=args.stride).items():
            all_samples.setdefault(face_id, []).extend(pts)

    result = frequency_separate(mesh, all_samples)
    label, message = verdict(result)

    print(f"Coverage: {result['coverage']:.1%} of faces sampled")
    print(f"Resolvable disagreement — median {result['median_resolvable']:.2f}°, "
          f"p90 {result['p90_resolvable']:.2f}°")
    print(f"Sub-face spread (unresolvable) — median {result['median_subface']:.2f}°")
    print(f"VERDICT: {label} — {message}")

    export_diagnostic_ply(mesh, result, args.out_ply)
    print(f"Heatmap: {args.out_ply} — red = likely real shape error, "
          f"gray = no PS coverage there.")


if __name__ == "__main__":
    main()