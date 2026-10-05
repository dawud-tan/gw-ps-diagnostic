"""Synthetic dressed mannequin for garment face selection (stage_garment.py), in the turntable frame.

- Mannequin: an elliptic body (0.16 x 0.11 m semi-axes, z 0 to 0.62 m) with a shoulder cap, a neck
  (r 0.045 m, to 0.74 m) and one arm stub (r 0.035 m, sticking out at +x, so the mannequin has no
  rotational symmetry), on the turntable top (r 0.25 m, z -0.03 to 0).
- Garment: the fold-textured torso of gwps.synth as a tube from z 0.03 to 0.59 m, 5-20 mm outside
  the body. A real scan hides the body under it, so the scan mesh holds the garment, the exposed
  mannequin (the body strips below and above the garment, the shoulders, the neck, the arm) and the
  turntable top, plus two floaters: one far out, one small and near.
- Reference: the bare mannequin and turntable top, moved by a rigid offset as if from another session.
"""
from __future__ import annotations

import numpy as np
import trimesh

from .synth import Rz, torso_mesh

BODY_A, BODY_B = 0.160, 0.110


def _body(z0, z1, n_theta=240, cap_top=False):
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    ring = np.stack([BODY_A * np.cos(th), BODY_B * np.sin(th)], -1)
    V = np.concatenate([np.c_[ring, np.full(n_theta, z0)], np.c_[ring, np.full(n_theta, z1)]])
    i = np.arange(n_theta)
    j = (i + 1) % n_theta
    F = np.concatenate([np.stack([i, j, j + n_theta], -1), np.stack([i, j + n_theta, i + n_theta], -1)])
    m = trimesh.Trimesh(V, F, process=False)
    if cap_top:
        c = len(V)
        cap = trimesh.Trimesh(np.vstack([V, [0, 0, z1]]), np.stack([np.full(n_theta, c), i + n_theta, j + n_theta], -1),
                              process=False)
        m = trimesh.util.concatenate([m, cap])
    return m


def _cylinder(r, z0, z1, sections=96, axis="z", centre=(0, 0)):
    c = trimesh.creation.cylinder(radius=r, height=z1 - z0, sections=sections)
    if axis == "x":                                     # along +x from z0 to z1 (as x), centred at (y, z) = centre
        c.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0]))
        c.apply_translation([(z0 + z1) / 2, centre[0], centre[1]])
    else:
        c.apply_translation([centre[0], centre[1], (z0 + z1) / 2])
    return c


def mannequin_parts():
    return {"body_low": _body(0.0, 0.03), "body_high": _body(0.59, 0.62, cap_top=True),
            "neck": _cylinder(0.045, 0.62, 0.74), "arm": _cylinder(0.035, 0.12, 0.27, axis="x", centre=(0.0, 0.52)),
            "turntable": _cylinder(0.25, -0.03, 0.0, sections=180)}


def make_scene(n_theta=340, n_z=200, ref_rot_deg=25.0, ref_shift_m=(0.003, -0.002, 0.0015), ref_tilt_deg=0.2):
    """-> scan mesh, garment truth (bool per face), bare reference (moved), the reference's move
    (4x4, reference frame -> scan frame is its inverse)."""
    tube, side = torso_mesh(z0=0.03, z1=0.59, n_theta=n_theta, n_z=n_z)
    garment = tube.submesh([np.flatnonzero(side)], append=True)
    parts = mannequin_parts()
    far = trimesh.creation.icosphere(2, 0.02)
    far.apply_translation([0.9, 0.0, 0.4])
    near = trimesh.creation.icosphere(2, 0.004)
    near.apply_translation([0.30, 0.05, 0.30])
    pieces = [garment] + list(parts.values()) + [far, near]
    scan = trimesh.util.concatenate(pieces)
    truth = np.zeros(len(scan.faces), bool)
    truth[:len(garment.faces)] = True
    bare = trimesh.util.concatenate([_body(0.0, 0.62, cap_top=True), parts["neck"], parts["arm"], parts["turntable"]])
    T = np.eye(4)
    T[:3, :3] = trimesh.transformations.rotation_matrix(np.radians(ref_tilt_deg), [1, 0, 0])[:3, :3] @ Rz(ref_rot_deg)
    T[:3, 3] = ref_shift_m
    bare.apply_transform(T)
    return scan, truth, bare, T
