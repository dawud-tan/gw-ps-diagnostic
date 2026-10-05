"""C1. ChArUco board: corners land on the true projections in COLMAP convention (the
OpenCV CharucoDetector +0.5 px quirk is neutralised), and board poses are accurate."""
import cv2
import numpy as np

from gwps.calib_io import CalibManifest, detection_image
from gwps.calib_synth import PILOT_CAM
from gwps.charuco import BoardConfig, board_pose, detect, face_normal_cam, to_uint8


def _placements(pilot):
    d = pilot.data()
    man = CalibManifest(d / "calib_manifest.csv")
    cfg = BoardConfig.load(d / "board.json")
    t = pilot.truth(d)
    for pos in man.positions("fabric_board"):
        R, tt = np.array(t["board_poses"][pos]["R"]), np.array(t["board_poses"][pos]["t"])
        yield pos, detection_image(man.slot("fabric_board", pos)), cfg, R, tt


def test_corner_pixel_convention(pilot):
    ours, raw = [], []
    for _, img, cfg, R, t in _placements(pilot):
        det = detect(img, cfg)
        ours.append(det.uv - PILOT_CAM.project(cfg.corners()[det.ids] @ R.T + t))
        cc, ci, _, _ = cv2.aruco.CharucoDetector(cfg.board()).detectBoard(to_uint8(img))
        ids = ci.ravel()
        k = cfg.usable_corners()[ids]
        # OpenCV's documented convention = COLMAP - 0.5
        raw.append(cc.reshape(-1, 2)[k] - (PILOT_CAM.project(cfg.corners()[ids[k]] @ R.T + t) - 0.5))
    o, r = np.concatenate(ours), np.concatenate(raw)
    print(f"\n[C1] ours vs truth (COLMAP): mean signed {o.mean(0).round(4)} px, rms {np.sqrt((o ** 2).sum(1).mean()):.3f} px, n={len(o)}")
    print(f"[C1] raw CharucoDetector vs OpenCV convention: mean signed {r.mean(0).round(3)} px (the +0.5 quirk)")
    assert np.all(np.abs(o.mean(0)) < 0.02)
    assert np.sqrt((o ** 2).sum(1).mean()) < 0.15


def test_board_poses(pilot):
    worst_n, worst_t = 0.0, 0.0
    for pos, img, cfg, R, t in _placements(pilot):
        Re, te, st = board_pose(detect(img, cfg), cfg, PILOT_CAM)
        n_err = np.degrees(np.arccos(np.clip(face_normal_cam(Re) @ face_normal_cam(R), -1, 1)))
        t_err = np.linalg.norm(te - t) * 1000
        print(f"[C1] {pos}: {st['n_corners']} corners, rms {st['rms_px']:.3f} px, normal err {n_err:.4f} deg, t err {t_err:.3f} mm")
        worst_n, worst_t = max(worst_n, n_err), max(worst_t, t_err)
    assert worst_n < 0.1 and worst_t < 0.5
