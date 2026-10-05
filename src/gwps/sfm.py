"""The COLMAP chain between a garment session and stage 1: SfM on the developed SfM images with
object masks, alignment to the turntable frame in metres, its checks, and undistortion.

Frames. COLMAP reconstructs the garment's frame at an arbitrary scale; the camera appears to orbit.
Stage 5 (azimuths, consistency bins) and the data contracts need the turntable frame: metres,
origin on the turntable axis at the turntable top, +Z up the axis, and here step 0's camera centre
on -Y (as in the synthetic scenes).

Intrinsics. With the pilot's intrinsics (stage C, OPENCV, COLMAP pixel convention) COLMAP keeps
them fixed, so the SfM model lives in the same camera frame the lights were calibrated in.
Without them it self-calibrates SIMPLE_RADIAL (COLMAP's default) and the lights' frame can differ
by the principal-point error (a small rotation); prefer the pilot's.

Axis and scale.
- The SfM camera centres lie on a circle about the turntable axis: the circle's normal is the axis
  direction, its centre a point on the axis, its radius r_sfm the camera's distance from the axis
  in model units.
- The metric-board frames (ChArUco board upright on the turntable, several turntable angles) give
  board poses in metres. A small bundle adjustment fits one rotation axis to all of them in pixels
  (rotation angle per frame, the board's pose at angle 0, the axis); the camera's distance from
  that axis, d_m, sets the scale s = d_m / r_sfm.
- Checks: circle residuals (wobble, SfM errors), the board frames' residuals from rotations about
  one axis (a board that slipped), the angle between the two axis estimates in the camera frame,
  and, if given, a tape-measured camera-to-axis distance (within 1 %). The tape check is coarse:
  the camera centre is the lens's entrance pupil, not a mark on the body.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from .camera import OpenCVCamera, read_cameras_txt, read_distorted_cameras_txt
from .masks import feature_mask, read_mask, write_mask
from .undistort import undistort_maps


# ---------------------------------------------------------------- COLMAP
def write_feature_masks(mask_dir, names, out_dir, margin_px):
    """COLMAP feature masks (<out>/<name>.png, 255 = extract) from object masks, eroded by margin_px."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        cv2.imwrite(str(out_dir / f"{n}.png"), feature_mask(read_mask(Path(mask_dir) / n), margin_px))
    return out_dir


def run_sfm(image_dir, work, names, intrinsics: OpenCVCamera | None = None, feature_mask_dir=None,
            max_features=8192, seed=0, num_threads=-1, filter_stationary=True, self_calibration_model="SIMPLE_RADIAL"):
    """Features, exhaustive matching (COLMAP's stationary-match filter on unless told otherwise),
    incremental mapping. -> (the reconstruction with the most registered images, stats)."""
    import pycolmap
    work = Path(work)
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    db = work / "database.db"
    ro = pycolmap.ImageReaderOptions()
    if intrinsics is not None:
        ro.camera_model = "OPENCV"
        ro.camera_params = ",".join(repr(float(p)) for p in intrinsics.params[:8])
    else:
        ro.camera_model = self_calibration_model
    if feature_mask_dir is not None:
        ro.mask_path = str(feature_mask_dir)
    eo = pycolmap.FeatureExtractionOptions()
    eo.sift.max_num_features = max_features
    eo.num_threads = num_threads
    pycolmap.extract_features(db, str(image_dir), image_names=list(names), camera_mode=pycolmap.CameraMode.SINGLE,
                              reader_options=ro, extraction_options=eo, device=pycolmap.Device.cpu)
    mo = pycolmap.FeatureMatchingOptions()
    mo.num_threads = num_threads
    tv = pycolmap.TwoViewGeometryOptions()
    tv.filter_stationary_matches = filter_stationary     # static background that masks missed
    pycolmap.match_exhaustive(db, matching_options=mo, verification_options=tv, device=pycolmap.Device.cpu)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.random_seed = seed
    opts.num_threads = num_threads
    if intrinsics is not None:
        opts.ba_refine_focal_length = False
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params = False
        opts.mapper.abs_pose_refine_focal_length = False
        opts.mapper.abs_pose_refine_extra_params = False
    recs = pycolmap.incremental_mapping(db, str(image_dir), str(work / "sparse"), opts)
    if not recs:
        raise RuntimeError("COLMAP registered no model")
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    with pycolmap.Database.open(db) as d:
        n_kp = [d.num_keypoints_for_image(im.image_id) for im in d.read_all_images()]
    stats = {"n_images": len(names), "n_registered": int(rec.num_reg_images()), "n_models": len(recs),
             "n_points3D": int(rec.num_points3D()), "mean_reprojection_error_px": float(rec.compute_mean_reprojection_error()),
             "mean_track_length": float(rec.compute_mean_track_length()),
             "keypoints_per_image_median": float(np.median(n_kp)) if n_kp else 0.0,
             "unregistered": sorted(set(names) - {im.name for im in rec.images.values() if im.has_pose})}
    return rec, stats


# ---------------------------------------------------------------- geometry
def fit_circle_3d(P):
    """Circle through points P (N,3): -> centre, unit normal, radius, per-point residual."""
    P = np.asarray(P, float)
    m = P.mean(0)
    _, _, Vt = np.linalg.svd(P - m)
    n = Vt[2]
    e1, e2 = Vt[0], Vt[1]
    q = np.stack([(P - m) @ e1, (P - m) @ e2], -1)
    A = np.c_[2 * q, np.ones(len(q))]                                  # Kasa fit
    cx, cy, k = np.linalg.lstsq(A, (q ** 2).sum(1), rcond=None)[0]
    c0 = m + cx * e1 + cy * e2
    r0 = np.sqrt(k + cx ** 2 + cy ** 2)

    def res(x):
        c, nn, r = x[:3], x[3:6] / np.linalg.norm(x[3:6]), x[6]
        d = P - c
        h = d @ nn
        rho = np.linalg.norm(d - h[:, None] * nn, axis=1)
        return np.hypot(rho - r, h)

    x = least_squares(res, np.r_[c0, n, r0], x_scale="jac").x
    c, n, r = x[:3], x[3:6] / np.linalg.norm(x[3:6]), float(x[6])
    return c, n, r, res(x)


def rot_about(a, theta):
    return cv2.Rodrigues(np.asarray(a, float) * theta)[0]


def fit_board_axis(detections, poses, board_corners, cam):
    """One rotation axis for several board poses (board -> camera, metres) of a board fixed on the
    turntable. Fits, in pixels: the axis (unit a, point c in the camera frame), the board's pose at
    angle 0, and an angle per frame (frame 0 at 0). -> dict with a, c, angles (deg), pixel RMS of
    the axis model and of the free per-frame poses, and the 3-D corner residual RMS (mm)."""
    Rs = [np.asarray(R) for R, _ in poses]
    ts = [np.asarray(t) for _, t in poses]
    # initial axis direction: rotation vectors of the relative rotations to frame 0, signs aligned
    rv = [cv2.Rodrigues(R @ Rs[0].T)[0].ravel() for R in Rs[1:]]
    big = max(rv, key=np.linalg.norm)
    a0 = big / np.linalg.norm(big)
    th0 = [0.0] + [float(np.dot(v, a0)) for v in rv]
    # initial axis point: t_k - c = Rot_k (t_0 - c)  ->  (I - Rot_k) c = t_k - Rot_k t_0
    A = np.concatenate([np.eye(3) - rot_about(a0, th) for th in th0[1:]] + [a0[None]])
    b = np.concatenate([ts[k] - rot_about(a0, th0[k]) @ ts[0] for k in range(1, len(ts))] + [np.zeros(1)])
    c0 = np.linalg.lstsq(A, b, rcond=None)[0]
    ref0 = np.r_[cv2.Rodrigues(Rs[0])[0].ravel(), ts[0]]
    obj = [board_corners[d.ids] for d in detections]
    uv = [d.uv for d in detections]

    def unpack(x):
        a = x[:3] / np.linalg.norm(x[:3])
        c = x[3:6]
        R_ref, t_ref = cv2.Rodrigues(x[6:9])[0], x[9:12]
        th = np.r_[0.0, x[12:]]
        return a, c, R_ref, t_ref, th

    def corners_cam(x, k):
        a, c, R_ref, t_ref, th = unpack(x)
        return (obj[k] @ R_ref.T + t_ref - c) @ rot_about(a, th[k]).T + c

    def res(x):
        return np.concatenate([(cam.project(corners_cam(x, k)) - uv[k]).ravel() for k in range(len(obj))])

    x = least_squares(res, np.r_[a0, c0, ref0, th0[1:]], x_scale="jac", method="lm").x
    a, c, _, _, th = unpack(x)
    c = c - (c @ a) * a                                               # the axis point nearest the camera
    r = res(x).reshape(-1, 2)
    free = np.concatenate([(cam.project(o @ R.T + t) - q) for o, q, R, t in zip(obj, uv, Rs, ts)])
    d3 = np.concatenate([corners_cam(x, k) - (obj[k] @ Rs[k].T + ts[k]) for k in range(len(obj))])
    return {"axis": a, "point": c, "angles_deg": np.degrees(th).tolist(),
            "rms_px": float(np.sqrt((r ** 2).sum(1).mean())), "free_rms_px": float(np.sqrt((free ** 2).sum(1).mean())),
            "corner_rms_mm": float(1000 * np.sqrt((d3 ** 2).sum(1).mean())),
            "camera_axis_distance_m": float(np.linalg.norm(c))}


def turntable_frame(centres, cam_up_world, first_centre):
    """Circle of camera centres (model units) -> origin (circle centre), unit axis pointing up,
    radius, residuals, and the new frame's rows e_x, e_y, e_z (step 0's camera on -Y)."""
    c, n, r, res = fit_circle_3d(centres)
    if np.dot(n, np.mean(cam_up_world, 0)) < 0:
        n = -n
    d0 = first_centre - c
    d0 = d0 - (d0 @ n) * n
    e_y = -d0 / np.linalg.norm(d0)
    e_x = np.cross(e_y, n)
    return {"centre": c, "axis": n, "radius": r, "residuals": res, "R": np.stack([e_x, e_y, n])}


def align_to_turntable(rec, board_fit=None, camera_height_m=None, scale=None):
    """Transform `rec` (pycolmap.Reconstruction, in place) to the turntable frame. The scale comes
    from board_fit (camera-to-axis distance in metres over the circle radius) unless given.
    -> report dict."""
    import pycolmap
    ims = sorted((im for im in rec.images.values() if im.has_pose), key=lambda im: im.name)
    Rw = [im.cam_from_world().rotation.matrix() for im in ims]
    C = np.array([im.projection_center() for im in ims])
    up = np.array([R.T @ np.array([0, -1.0, 0]) for R in Rw])       # camera 'up' in the model
    tt = turntable_frame(C, up, C[0])
    s = scale if scale is not None else (board_fit["camera_axis_distance_m"] / tt["radius"] if board_fit else None)
    if s is None:
        raise ValueError("no metric scale: give the metric-board fit or a scale")
    origin = tt["centre"] - ((camera_height_m / s) * tt["axis"] if camera_height_m is not None else 0)
    R = tt["R"]
    rec.transform(pycolmap.Sim3d(float(s), pycolmap.Rotation3d(R), -s * R @ origin))
    az = np.degrees(np.arctan2(*(((C - tt["centre"]) @ R.T)[:, [1, 0]].T)))
    steps = np.diff(np.unwrap(np.radians(az)))
    rep = {"scale_m_per_unit": float(s), "circle_radius_m": float(s * tt["radius"]),
           "circle_rms_mm": float(1000 * s * np.sqrt(np.mean(tt["residuals"] ** 2))),
           "circle_max_mm": float(1000 * s * tt["residuals"].max()),
           "camera_height_m": camera_height_m,
           "camera_azimuth_deg": {im.name: float(a) for im, a in zip(ims, az)},
           "step_deg": {"median": float(np.degrees(np.median(np.abs(steps)))) if len(steps) else None,
                        "min": float(np.degrees(np.abs(steps).min())) if len(steps) else None,
                        "max": float(np.degrees(np.abs(steps).max())) if len(steps) else None}}
    if board_fit is not None:
        a_cam = np.array([Ri @ tt["axis"] for Ri in Rw]).mean(0)   # the SfM axis in the camera frame
        a_cam /= np.linalg.norm(a_cam)
        rep["axis_angle_to_board_deg"] = float(np.degrees(np.arccos(min(1.0, abs(float(a_cam @ board_fit["axis"]))))))
    return rep


# ---------------------------------------------------------------- undistortion
def undistort(model_dir, image_dir, out_dir, names, mask_dir=None):
    """COLMAP's image_undistorter on the aligned model, its sparse model rewritten as text, and the
    object masks undistorted with the same mapping (bilinear, then > 0.5)."""
    import pycolmap
    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    pycolmap.undistort_images(str(out_dir), str(model_dir), str(image_dir), image_names=list(names))
    sp = out_dir / "sparse"
    pycolmap.Reconstruction(str(sp)).write_text(str(sp))
    for f in ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin"):
        (sp / f).unlink(missing_ok=True)
    if mask_dir is not None:
        src = read_distorted_cameras_txt(Path(model_dir) / "cameras.txt")
        dst = read_cameras_txt(sp / "cameras.txt")
        maps = {cid: undistort_maps(src[cid], dst[cid]) for cid in dst}
        from .camera import read_images_txt
        for im in read_images_txt(sp / "images.txt").values():
            m = read_mask(Path(mask_dir) / im.name).astype(np.float32)
            m1, m2 = maps[im.camera_id]
            u = cv2.remap(m, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            write_mask(out_dir / "masks" / im.name, u > 0.5)
    return out_dir
