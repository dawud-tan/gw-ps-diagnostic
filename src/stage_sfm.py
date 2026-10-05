"""COLMAP chain for a garment session: SfM with object masks -> turntable frame in metres ->
checks -> undistorted images, masks and lit frames for stage 1 (see gwps/sfm.py).

  stage_sfm.py --session captures/shirt01 --intrinsics captures/pilot/intrinsics.json \
               --board config/board.json --camera-height-m 0.30 --out runs/shirt01/sfm

Reads the session's colmap_images/ (capture_session.py --develop), masks/ (stage_masks.py; or
--no-masks), calib_manifest.csv (the metric_board frames) and manifest.csv (the image names, and
the lit frames). --camera-height-m: the lens axis above the turntable top (tape, +-5 mm), which
puts the origin on the turntable top; without it the origin is at the camera's height.
Writes:
  <out>/colmap/        database and COLMAP's model(s)
  <out>/aligned/       the model in the turntable frame (text, distorted camera)
  <out>/undistorted/   images/, sparse/ (text, PINHOLE), masks/   <- stage 1's --undistorted, --masks
  <out>/lit/           every manifest frame undistorted to that camera   <- stage 1's --lit-manifest
  <out>/sfm.json, sfm.md
Gates (exit code 1 if any fails): every image registered; COLMAP's mean reprojection error <= 1 px;
camera centres within 3 mm RMS of one circle; the metric-board frames fit one rotation axis within
1 mm RMS; the SfM and board axes within 0.5 deg; the tape-measured camera-to-axis distance (if given)
within 1 %.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import undistort_lit
from gwps import sfm as S
from gwps.calib_io import CalibManifest, detection_image
from gwps.camera import OpenCVCamera, load_camera_model
from gwps.charuco import BoardConfig, board_pose, detect
from gwps.io import Manifest, write_meta
from gwps.masks import default_feature_margin_px

GATES = {"reprojection_px": 1.0, "circle_rms_mm": 3.0, "board_axis_rms_mm": 1.0, "axis_angle_deg": 0.5,
         "measured_distance_rel": 0.01}


def metric_board_fit(calib_manifest, board, cam, camera_id=1):
    """Board poses from the session's metric_board frames and one rotation axis through them."""
    man = CalibManifest(calib_manifest)
    cfg = BoardConfig.load(board)
    dets, poses, per = [], [], {}
    for pos in man.positions("metric_board", camera_id):
        det = detect(detection_image(man.slot("metric_board", pos)), cfg)
        if len(det.ids) < 12:
            per[pos] = {"n_corners": int(len(det.ids)), "used": False}
            continue
        R, t, st = board_pose(det, cfg, cam)
        dets.append(det)
        poses.append((R, t))
        per[pos] = {**st, "used": True}
    if len(poses) < 2:
        raise ValueError(f"only {len(poses)} metric-board frame(s) usable: need >= 2 at different turntable angles")
    fit = S.fit_board_axis(dets, poses, cfg.corners(), cam)
    fit["frames"] = per
    return fit


def run(session, intrinsics, board, out, masks="auto", camera_height_m=None, measured_axis_distance_m=None,
        max_features=8192, seed=0, lit=True, gates=None, filter_stationary=True, self_calibration_model="SIMPLE_RADIAL"):
    """intrinsics: the pilot's intrinsics JSON (recommended), or None to let COLMAP self-calibrate
    `self_calibration_model` (its scale was 2.5 % off on the synthetic lens with SIMPLE_RADIAL;
    the board gate caught it)."""
    session, out = Path(session), Path(out)
    gates = {**GATES, **(gates or {})}
    man = Manifest(session / "manifest.csv")
    names = sorted(man.views)
    image_dir = session / "colmap_images"
    missing = [n for n in names if not (image_dir / n).exists()]
    if missing:
        raise ValueError(f"{len(missing)} SfM image(s) missing in {image_dir} (run capture_session.py --develop): {missing[:3]}")
    cam = load_camera_model(intrinsics) if intrinsics else None
    mask_dir = (session / "masks" if (session / "masks").exists() else None) if masks == "auto" else (Path(masks) if masks else None)
    rep = {"session": str(session), "intrinsics": str(intrinsics) if intrinsics else f"self-calibrated ({self_calibration_model})",
           "masks": str(mask_dir) if mask_dir else None}
    fmask = None
    if mask_dir is not None:
        import cv2
        w = cv2.imread(str(image_dir / names[0]), cv2.IMREAD_GRAYSCALE).shape[1]
        margin = default_feature_margin_px(w)
        fmask = S.write_feature_masks(mask_dir, names, out / "colmap_masks", margin)
        rep["feature_mask_margin_px"] = margin
    cman = session / "calib_manifest.csv"
    if not cman.exists() or not CalibManifest(cman).positions("metric_board"):
        raise ValueError(f"{session}: no metric_board frames in calib_manifest.csv; the scale needs them")
    rec, rep["sfm"] = S.run_sfm(image_dir, out / "colmap", names, cam, fmask, max_features, seed,
                                filter_stationary=filter_stationary, self_calibration_model=self_calibration_model)
    if cam is None:                                   # self-calibrated: fit the board with COLMAP's camera
        c = next(iter(rec.cameras.values()))
        cam = OpenCVCamera.from_colmap(1, c.model_name, c.width, c.height, list(c.params))
    fit = metric_board_fit(cman, board, cam)
    rep["metric_board"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fit.items()}
    rep["alignment"] = S.align_to_turntable(rec, fit, camera_height_m)
    aligned = out / "aligned"
    aligned.mkdir(parents=True, exist_ok=True)
    rec.write_text(str(aligned))
    und = S.undistort(aligned, image_dir, out / "undistorted", names, mask_dir)
    if lit:
        m, c = undistort_lit.run(session / "manifest.csv", aligned / "cameras.txt", und / "sparse/cameras.txt", out / "lit")
        rep["lit_manifest"] = str(m)
    # gates
    g = {"all_registered": rep["sfm"]["n_registered"] == len(names),
         "reprojection": rep["sfm"]["mean_reprojection_error_px"] <= gates["reprojection_px"],
         "circle": rep["alignment"]["circle_rms_mm"] <= gates["circle_rms_mm"]}
    g["board_axis"] = fit["corner_rms_mm"] <= gates["board_axis_rms_mm"]
    g["axis_angle"] = rep["alignment"]["axis_angle_to_board_deg"] <= gates["axis_angle_deg"]
    if measured_axis_distance_m:
        d = fit["camera_axis_distance_m"]
        rep["measured_axis_distance"] = {"tape_m": measured_axis_distance_m, "fitted_m": d,
                                         "relative": abs(d - measured_axis_distance_m) / measured_axis_distance_m}
        g["measured_distance"] = rep["measured_axis_distance"]["relative"] <= gates["measured_distance_rel"]
    rep["gates"] = g
    rep["pass"] = bool(all(g.values()))
    rep["tolerances"] = gates
    (out / "sfm.json").write_text(json.dumps(rep, indent=1, default=float))
    (out / "sfm.md").write_text(markdown(rep))
    write_meta(out, {"session": session, "intrinsics": intrinsics, "board": board, "masks": mask_dir},
               {"max_features": max_features, "seed": seed, "camera_height_m": camera_height_m})
    return rep


def markdown(rep):
    s, a = rep["sfm"], rep["alignment"]
    L = [f"# SfM and turntable alignment: {'PASS' if rep['pass'] else '**FAIL**'}", "",
         f"- Registered {s['n_registered']} / {s['n_images']} images ({s['n_models']} model(s)); "
         f"{s['n_points3D']} points; mean reprojection error {s['mean_reprojection_error_px']:.3f} px; "
         f"mean track length {s['mean_track_length']:.1f}; keypoints per image (median) {s['keypoints_per_image_median']:.0f}.",
         f"- Intrinsics: {rep['intrinsics']}. Masks: {rep['masks'] or 'none'}.",
         f"- Scale {a['scale_m_per_unit']:.6g} m per model unit; camera-to-axis distance {a['circle_radius_m']:.4f} m; "
         f"camera centres off the circle {a['circle_rms_mm']:.2f} mm RMS (max {a['circle_max_mm']:.2f} mm).",
         f"- Turntable steps {a['step_deg']['median']:.2f} deg median ({a['step_deg']['min']:.2f}-{a['step_deg']['max']:.2f})."
         if a["step_deg"]["median"] is not None else "- One image only.",
         f"- Origin: {'turntable top, %.3f m below the camera' % a['camera_height_m'] if a['camera_height_m'] is not None else 'at the camera height (no --camera-height-m)'}."]
    if "metric_board" in rep:
        m = rep["metric_board"]
        L.append(f"- Metric board: {sum(v['used'] for v in m['frames'].values())} frames, one axis fits them to "
                 f"{m['rms_px']:.3f} px (free poses {m['free_rms_px']:.3f} px), corners {m['corner_rms_mm']:.3f} mm RMS; "
                 f"camera-to-axis {m['camera_axis_distance_m']:.4f} m; SfM axis vs board axis "
                 f"{a['axis_angle_to_board_deg']:.3f} deg.")
    if "measured_axis_distance" in rep:
        d = rep["measured_axis_distance"]
        L.append(f"- Tape: {d['tape_m']:.4f} m against {d['fitted_m']:.4f} m ({100 * d['relative']:.2f} %).")
    L += ["", "| gate | result |", "|---|---|"] + [f"| {k} | {'ok' if v else '**FAIL**'} |" for k, v in rep["gates"].items()]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--intrinsics", default=None, help="pilot intrinsics JSON (stage C, OPENCV); required unless --self-calibrate")
    ap.add_argument("--self-calibrate", default=None, choices=["SIMPLE_RADIAL", "RADIAL", "OPENCV"],
                    help="dry runs only: let COLMAP self-calibrate this model (the principal point stays at the image "
                         "centre; the scale was 2.5 %% off on the synthetic lens with SIMPLE_RADIAL)")
    ap.add_argument("--board", required=True, help="board.json of the metric-board frames")
    ap.add_argument("--out", required=True)
    ap.add_argument("--masks", default="auto", help="object masks dir (default <session>/masks if present)")
    ap.add_argument("--no-masks", action="store_true")
    ap.add_argument("--camera-height-m", type=float, default=None)
    ap.add_argument("--measured-axis-distance-m", type=float, default=None,
                    help="tape: camera's entrance pupil to the turntable axis (checked within 1 %%)")
    ap.add_argument("--max-features", type=int, default=8192)
    ap.add_argument("--no-lit", action="store_true", help="skip undistorting the lit frames")
    a = ap.parse_args()
    if bool(a.intrinsics) == bool(a.self_calibrate):
        ap.error("give --intrinsics (the pilot's) or, for a dry run, --self-calibrate MODEL")
    r = run(a.session, a.intrinsics, a.board, a.out, None if a.no_masks else a.masks, a.camera_height_m,
            a.measured_axis_distance_m, a.max_features, lit=not a.no_lit,
            self_calibration_model=a.self_calibrate or "SIMPLE_RADIAL")
    print((Path(a.out) / "sfm.md").read_text())
    sys.exit(0 if r["pass"] else 1)
