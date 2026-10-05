"""Control datasets for stage 6, in the same layout the stage 3-5 code reads for the garment.

Board: world = board frame, one "image" per placement with pose board -> camera from the
ChArUco corners (like the turntable: object fixed, camera moves). The mesh is only the
fabric rectangle, offset by the fabric thickness toward the camera (-z).
Sphere: world = sphere frame (origin at the centre, camera axes), pose R = I, t = centre.
Lights stay in the camera frame, so the calibrated lights.json is used unchanged.
Images must be PINHOLE (undistorted) for the per-pixel PS in stages 4-5.
"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path

import numpy as np
import trimesh

from .camera import Image, R_to_qvec, write_colmap_txt


def _write(out, cam, poses, slots, mesh, lights_json):
    """poses: name -> (R, t); slots: name -> manifest slot (lights, ambient)."""
    out = Path(out)
    images = {}
    rows = []
    for k, (name, (R, t)) in enumerate(poses.items()):
        img_name = f"{name}.png"
        images[k + 1] = Image(k + 1, R_to_qvec(R), np.asarray(t, float), cam.camera_id, img_name)
        slot = slots[name]
        for lid, p in sorted(slot["lights"].items()):
            rows.append([k, cam.camera_id, img_name, lid, str(Path(p).resolve())])
        if "ambient" in slot:
            rows.append([k, cam.camera_id, img_name, "ambient", str(Path(slot["ambient"]).resolve())])
    write_colmap_txt(out / "sparse" / "0", {cam.camera_id: cam}, images)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "camera_id", "colmap_image_name", "light_id", "path"])
        w.writerows(rows)
    mesh.export(out / "mesh.ply")
    np.save(out / "garment_faces.npy", np.ones(len(mesh.faces), bool))
    shutil.copy(lights_json, out / "lights.json")
    return out


def fabric_mesh(cfg, n=(80, 60)):
    x0, y0, x1, y1 = cfg.fabric_region_m
    xs, ys = np.linspace(x0, x1, n[0]), np.linspace(y0, y1, n[1])
    X, Y = np.meshgrid(xs, ys)
    V = np.stack([X, Y, np.full_like(X, -cfg.fabric_thickness_m)], -1).reshape(-1, 3)
    idx = np.arange(n[0] * n[1]).reshape(n[1], n[0])
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    F = np.concatenate([np.stack([a, c, d], -1).reshape(-1, 3), np.stack([a, d, b], -1).reshape(-1, 3)])
    m = trimesh.Trimesh(V, F, process=False)
    if m.face_normals[:, 2].mean() > 0:            # printed face points along -z
        m = trimesh.Trimesh(V, F[:, ::-1], process=False)
    return m


def build_board_control(out, cam, cfg, poses, slots, lights_json):
    if cfg.fabric_region_m is None:
        raise ValueError("board.json has no fabric_region_m")
    return _write(out, cam, poses, slots, fabric_mesh(cfg), lights_json)


def build_sphere_control(out, cam, radius, centres, slots, lights_json, subdivisions=7):
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    poses = {name: (np.eye(3), C) for name, C in centres.items()}
    return _write(out, cam, poses, slots, mesh, lights_json)
