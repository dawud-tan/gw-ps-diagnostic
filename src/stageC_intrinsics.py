"""Stage C: camera intrinsics (COLMAP OPENCV model, COLMAP pixel convention) from
ChArUco shots listed under target 'intrinsics' in config/calib_manifest.csv."""
import argparse
import json
from pathlib import Path

import numpy as np

from gwps.calib_io import CalibManifest, detection_image
from gwps.camera import OpenCVCamera
from gwps.charuco import BoardConfig, board_pose, calibrate_intrinsics, detect
from gwps.io import load_linear


def run(calib_manifest, board, out, camera_id=1, min_corners=12):
    man = CalibManifest(calib_manifest)
    cfg = BoardConfig.load(board)
    dets, used, H, W = [], [], None, None
    for pos in man.positions("intrinsics", camera_id):
        img = detection_image(man.slot("intrinsics", pos))
        H, W = img.shape
        d = detect(img, cfg)
        if len(d.ids) >= min_corners:
            dets.append(d)
            used.append(pos)
    if len(dets) < 6:
        raise ValueError(f"only {len(dets)} usable intrinsics views (need >= 6, ideally ~15 covering the frame)")
    params, rms = calibrate_intrinsics(dets, cfg, W, H)
    cam = OpenCVCamera(camera_id, W, H, params)
    per_view = {}
    for pos, d in zip(used, dets):
        _, _, st = board_pose(d, cfg, cam)
        per_view[pos] = {"n_corners": st["n_corners"], "rms_px": st["rms_px"]}
    allc = np.concatenate([d.uv for d in dets])
    cover = [float(allc[:, 0].min() / W), float(allc[:, 0].max() / W), float(allc[:, 1].min() / H), float(allc[:, 1].max() / H)]
    info = cam.to_json({"rms_px": rms, "n_views": len(dets), "views": per_view,
                        "corner_coverage_u_min_max_v_min_max": cover})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(info, indent=1))
    return cam, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-manifest", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera-id", type=int, default=1)
    a = ap.parse_args()
    _, info = run(a.calib_manifest, a.board, a.out, a.camera_id)
    print(json.dumps({k: info[k] for k in ("params", "rms_px", "n_views", "corner_coverage_u_min_max_v_min_max")}, indent=1))
