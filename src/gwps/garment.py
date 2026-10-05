"""Which faces of the raw GW mesh are the garment (config/garment_faces.npy): the mannequin, the
turntable top and any floaters must stay out of every statistic and out of the bake.

The mesh must be in the turntable frame (stage_sfm.py: metres, origin on the axis at the turntable
top, +Z up). Criteria, in order; each is optional except the region:

1. Region: z >= z_min above the turntable top (default 1 cm, drops the top itself), z <= z_max if
   given, and within r_max of the axis (drops floaters and stray background).
2. Bare-mannequin reference: a mesh of the same mannequin on the same turntable without the garment,
   from its own capture (stage_sfm + GW, once per mannequin). Faces within ref_dist (default 2 mm)
   of it are mannequin or turntable. The reference is registered to the scan first: the mannequin
   may sit at another angle, and each session's frame fixes its azimuth by its own step 0. A coarse
   search over rotation about the axis is followed by trimmed ICP, whose tolerance shrinks until
   only surfaces the two scans share (the turntable top, exposed mannequin parts) pull on it; the
   garment-covered body, a few mm or more under the garment, drops out.
   A garment that fits within ref_dist of the mannequin's surface (tight knits) is lost there too:
   it only leaves the statistics, it does not make a false cluster.
3. Per-view garment masks (255 = garment, 0 = anything else; painted, or from a segmenter), in the
   geometry of stage 3's views: each face takes the majority of the labelled pixels that see it
   (stage 3's face-id maps). Faces no labelled view sees keep the other criteria's call.
4. Cleanup: selected pieces smaller than min_component_frac of the selection's area are dropped.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def region(mesh, z_min=0.01, z_max=None, r_max=0.6):
    c = mesh.triangles_center
    keep = (c[:, 2] >= z_min) & (np.hypot(c[:, 0], c[:, 1]) <= r_max)
    if z_max is not None:
        keep &= c[:, 2] <= z_max
    return keep


def surface_points(mesh, spacing_m=0.001, seed=0, max_points=4_000_000):
    """Random points on the surface at ~one per spacing^2, plus the vertices."""
    import trimesh
    n = int(min(max_points, max(1000, mesh.area / spacing_m ** 2)))
    pts, _ = trimesh.sample.sample_surface(mesh, n, seed=seed)
    return np.vstack([pts, mesh.vertices])


def _kabsch(P, Q):
    """Rigid (R, t) minimising |R P + t - Q|."""
    mp, mq = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    return R, mq - R @ mp


def _rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])


def _icp(src, tree, R, t, taus, iters=8):
    """Trimmed ICP of src points onto the tree's points, from (R, t), shrinking the tolerance."""
    for tau in taus:
        for _ in range(iters):
            moved = src @ R.T + t
            d, j = tree.query(moved, distance_upper_bound=tau, workers=-1)
            ok = np.isfinite(d)
            if ok.sum() < 50:
                break
            dR, dt = _kabsch(moved[ok], tree.data[j[ok]])
            R, t = dR @ R, dR @ t + dt
    moved = src @ R.T + t
    d, _ = tree.query(moved, distance_upper_bound=taus[-1], workers=-1)
    return R, t, float(np.isfinite(d).mean())


def register_reference(ref_mesh, scan_mesh, n_points=60_000, seed=0, step_deg=2.0, n_peaks=4):
    """Rigid transform (R, t) taking the reference onto the scan. Both are in turntable frames, so
    they differ by a rotation about +Z (how the mannequin sat) and a few mm (placement, the tape
    measurement of camera height that sets each frame's origin).

    Two traps, both measured on the synthetic scene: a loose inlier score favours wrong angles,
    because the reference's body, hidden under the garment in the scan, crosses the garment surface
    when misrotated (at 5 mm, no rotation outscored the true one); and a tight score of rotation
    alone is noise, because a few mm of offset already pushes the shared surfaces past it (the true
    angle scored 0.28 at 2 mm, against 0.53 with its offset). So from every step_deg a short trimmed
    ICP (5 -> 2 mm) fits rotation and offset together on a subsample and is scored at 2 mm; the best
    few separated starts (an elliptic body has a half-turn twin) are refined on all points down to
    1 mm, and the best fit at 1 mm wins. -> R, t, report."""
    import trimesh
    rng = np.random.default_rng(seed)
    src, _ = trimesh.sample.sample_surface(ref_mesh, n_points, seed=seed)
    dst, _ = trimesh.sample.sample_surface(scan_mesh, 4 * n_points, seed=seed + 1)
    tree = cKDTree(dst)
    sub = src[rng.choice(len(src), min(len(src), 8_000), replace=False)]
    starts = []
    for a in np.arange(0.0, 360.0, step_deg):
        R, t, frac = _icp(sub, tree, _rz(a), np.zeros(3), (0.005, 0.003, 0.002), iters=3)
        starts.append((frac, a, R, t))
    starts.sort(key=lambda s: -s[0])
    peaks = []
    for s in starts:
        if all(abs((s[1] - p[1] + 180) % 360 - 180) > 10 for p in peaks):
            peaks.append(s)
        if len(peaks) == n_peaks:
            break
    fits = [_icp(src, tree, R0, t0, (0.003, 0.002, 0.0015, 0.001), iters=10) + (a,) for _, a, R0, t0 in peaks]
    fits.sort(key=lambda f: -f[2])
    R, t, frac, a = fits[0]
    ang = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    return R, t, {"rotation_deg": ang, "translation_mm": (1000 * t).tolist(), "inlier_fraction_1mm": frac,
                  "runner_up_inlier_fraction_1mm": fits[1][2] if len(fits) > 1 else None,
                  "search_deg": float(a), "R": R.tolist(), "t_m": t.tolist()}


def near_reference(scan_mesh, ref_mesh, dist_m=0.002, R=None, t=None, spacing_m=0.001):
    """Faces of the scan whose centre lies within dist_m of the (transformed) reference surface."""
    pts = surface_points(ref_mesh, spacing_m)
    if R is not None:
        pts = pts @ R.T + t
    d, _ = cKDTree(pts).query(scan_mesh.triangles_center, distance_upper_bound=dist_m + spacing_m, workers=-1)
    return d <= dist_m + 0.5 * spacing_m, d


def vote(face_id_maps, masks, n_faces):
    """Per-face counts of garment and labelled pixels over the labelled views.
    face_id_maps, masks: {view name: (H, W) array}; face ids < 0 are misses."""
    g = np.zeros(n_faces, np.int64)
    n = np.zeros(n_faces, np.int64)
    for name, m in masks.items():
        fid = face_id_maps[name]
        if fid.shape != m.shape:
            raise ValueError(f"{name}: garment mask {m.shape} does not match stage 3's view {fid.shape}")
        hit = fid >= 0
        n += np.bincount(fid[hit], minlength=n_faces)
        g += np.bincount(fid[hit & m], minlength=n_faces)
    return g, n


def drop_small_pieces(mesh, sel, min_frac=0.01):
    """Keep selected connected pieces (over shared edges) of at least min_frac of the selection's area."""
    if not sel.any():
        return sel
    adj = mesh.face_adjacency
    both = sel[adj[:, 0]] & sel[adj[:, 1]]
    a = adj[both]
    F = len(mesh.faces)
    _, lab = connected_components(coo_matrix((np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(F, F)), directed=False)
    area = np.bincount(lab[sel], weights=mesh.area_faces[sel], minlength=lab.max() + 1)
    big = area >= min_frac * area.sum()
    return sel & big[lab]


def select(mesh, z_min=0.01, z_max=None, r_max=0.6, reference=None, ref_dist_m=0.002, register=True,
           face_id_maps=None, masks=None, min_component_frac=0.01):
    """-> bool per face (garment), report."""
    area = mesh.area_faces
    rep = {"faces": int(len(mesh.faces)), "area_m2": float(area.sum())}
    sel = region(mesh, z_min, z_max, r_max)
    rep["region"] = {"z_min": z_min, "z_max": z_max, "r_max": r_max, "faces": int(sel.sum()),
                     "area_m2": float(area[sel].sum())}
    if reference is not None:
        R = t = None
        if register:
            R, t, rep["registration"] = register_reference(reference, mesh)
        near, _ = near_reference(mesh, reference, ref_dist_m, R, t)
        sel &= ~near
        rep["reference"] = {"dist_mm": 1000 * ref_dist_m, "faces_near": int(near.sum()),
                            "area_near_m2": float(area[near].sum())}
    if masks:
        g, n = vote(face_id_maps, masks, len(mesh.faces))
        seen = n > 0
        sel = np.where(seen, g * 2 > n, sel)
        rep["masks"] = {"views": len(masks), "faces_labelled": int(seen.sum()),
                        "area_labelled_m2": float(area[seen].sum())}
    before = sel.copy()
    sel = drop_small_pieces(mesh, sel, min_component_frac)
    rep["cleanup"] = {"min_component_frac": min_component_frac, "faces_dropped": int((before & ~sel).sum())}
    rep["garment"] = {"faces": int(sel.sum()), "area_m2": float(area[sel].sum())}
    return sel, rep
