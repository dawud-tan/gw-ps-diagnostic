"""Embree ray casting. The intersector is constructed explicitly so a missing embreex
fails loudly instead of falling back to trimesh's pure-Python intersector (which
exhausts memory at under 10k rays against a 5k-face mesh)."""
from __future__ import annotations

import numpy as np
import trimesh
from trimesh.ray.ray_pyembree import RayMeshIntersector

BATCH = 1_000_000


class Caster:
    def __init__(self, mesh: trimesh.Trimesh):
        self.mesh = mesh
        self.rmi = RayMeshIntersector(mesh)

    def first_hit(self, origins, dirs):
        """-> face id (N,) with -1 for a miss, hit locations (N,3), distance along ray (N,)."""
        origins = np.broadcast_to(np.asarray(origins, np.float64), np.shape(dirs))
        n = len(dirs)
        fid = np.full(n, -1, np.int64)
        loc = np.full((n, 3), np.nan)
        for s in range(0, n, BATCH):
            o, d = origins[s:s + BATCH], dirs[s:s + BATCH]
            tri, ray, pts = self.rmi.intersects_id(o, d, multiple_hits=False, return_locations=True)
            fid[s + ray] = tri
            loc[s + ray] = pts
        dist = np.einsum("ij,ij->i", loc - origins, dirs)
        return fid, loc, dist

    def occluded(self, origins, targets, eps=1e-4):
        """True where the segment origin -> target is blocked (origin offset by eps along it)."""
        v = targets - origins
        r = np.linalg.norm(v, axis=1)
        d = v / r[:, None]
        fid, _, dist = self.first_hit(origins + eps * d, d)
        return (fid >= 0) & (dist < r - 2 * eps)


def cast_view(caster, cam, R, t, mask_fn=None, max_rays=1_000_000):
    """Cast one ray per pixel centre, in row blocks of <= max_rays, into float32 camera-frame
    maps (row-major, H x W): hit, depth (planar z), pos_cam, face_id, normal_cam."""
    H, W = cam.height, cam.width
    C = -R.T @ t
    hit = np.zeros((H, W), bool)
    depth = np.zeros((H, W), np.float32)
    pos = np.zeros((H, W, 3), np.float32)
    fid_img = np.full((H, W), -1, np.int32)
    nrm = np.zeros((H, W, 3), np.float32)
    rows = max(1, max_rays // W)
    jj = np.arange(W) + 0.5
    for r0 in range(0, H, rows):
        r1 = min(H, r0 + rows)
        u, v = np.meshgrid(jj, np.arange(r0, r1) + 0.5)
        d_cam = cam.rays(np.stack([u.ravel(), v.ravel()], -1))
        fid, loc, _ = caster.first_hit(C, d_cam @ R)          # R^T d per row
        h = fid >= 0
        blk = (slice(r0, r1), slice(None))
        hit[blk] = h.reshape(r1 - r0, W)
        fid_img[blk] = fid.reshape(r1 - r0, W)
        p_cam = np.zeros((len(fid), 3))
        p_cam[h] = loc[h] @ R.T + t
        pos[blk] = p_cam.reshape(r1 - r0, W, 3)
        depth[blk] = p_cam[:, 2].reshape(r1 - r0, W)
        n_cam = np.zeros((len(fid), 3))
        n_cam[h] = caster.mesh.face_normals[fid[h]] @ R.T
        nrm[blk] = n_cam.reshape(r1 - r0, W, 3)
    return {"hit": hit, "depth": depth, "pos_cam": pos, "face_id": fid_img, "normal_cam": nrm}
