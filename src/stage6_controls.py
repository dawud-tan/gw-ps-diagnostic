"""Stage 6 on real-layout control captures: board poses from ChArUco, matte-ball centres
from silhouettes -> control datasets -> stages 3-5 -> floors (fabric and sphere).

The lit images must be undistorted to the PINHOLE camera given here. Note that the matte
ball is also what stage C fits E / axis / mu to, so the sphere floor is optimistic; the
verdict uses the fabric floor. Board placements ending in '_par' (lens polariser parallel)
give a separate 'fabric_parallel' floor: crossed versus parallel decides cross-polarisation.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import stage6_floor
from gwps.calib_io import CalibManifest, detection_image
from gwps.camera import Camera, load_camera_model
from gwps.charuco import BoardConfig, board_pose, detect
from gwps.controls import build_board_control, build_sphere_control
from stageC_lights import ball_centres


def run(calib_manifest, board, camera, lights, out, ball_radius_m=None, camera_id=1, poses_override=None,
        centres_override=None, read_noise=0.005, full_well=10000.0, noise=None, model_error=0.005):
    man = CalibManifest(calib_manifest)
    cam = load_camera_model(camera, camera_id)
    if not isinstance(cam, Camera):
        raise ValueError("stage 6 needs undistorted lit images and a PINHOLE camera (cameras.txt); "
                         "undistort the raw lit images first with undistort_lit.py")
    out = Path(out)
    floors_path = out / "floors.json"
    report = {}
    all_board = man.positions("fabric_board", camera_id)
    for kind, positions in (("fabric", [p for p in all_board if not p.endswith("_par")]),
                            ("fabric_parallel", [p for p in all_board if p.endswith("_par")])):
        if not positions:
            continue
        cfg = BoardConfig.load(board)
        poses, slots, pose_stats = {}, {}, {}
        for pos in positions:
            slot = man.slot("fabric_board", pos)
            if poses_override and pos in poses_override:
                poses[pos] = poses_override[pos]
            else:
                R, t, st = board_pose(detect(detection_image(slot), cfg), cfg, cam)
                poses[pos], pose_stats[pos] = (R, t), st
            slots[pos] = slot
        ds = build_board_control(out / kind, cam, cfg, poses, slots, lights)
        report[kind] = stage6_floor.run(ds / "sparse/0", ds / "mesh.ply", ds / "garment_faces.npy",
                                            ds / "manifest.csv", ds / "lights.json", ds / "run", kind,
                                            floors_path=floors_path, read_noise=read_noise, full_well=full_well,
                                            noise=noise, model_error=model_error)
        report[f"{kind}_board_poses"] = pose_stats
    if ball_radius_m and man.positions("matte_ball", camera_id):
        centres = centres_override or ball_centres(man, cam, ball_radius_m, camera_id)[0]
        slots = {p: man.slot("matte_ball", p) for p in centres}
        ds = build_sphere_control(out / "matte_ball", cam, ball_radius_m, centres, slots, lights)
        report["sphere"] = stage6_floor.run(ds / "sparse/0", ds / "mesh.ply", ds / "garment_faces.npy",
                                            ds / "manifest.csv", ds / "lights.json", ds / "run", "sphere",
                                            floors_path=floors_path, read_noise=read_noise, full_well=full_well,
                                            noise=noise, model_error=model_error)
    (out / "controls_report.json").write_text(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-manifest", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--camera", required=True, help="cameras.txt with the PINHOLE camera of the undistorted images")
    ap.add_argument("--camera-id", type=int, default=1)
    ap.add_argument("--lights", required=True)
    ap.add_argument("--ball-radius-m", type=float, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise", default=None, help="stage-C radiometry.json (measured noise model)")
    ap.add_argument("--model-error", type=float, default=0.005)
    a = ap.parse_args()
    rep = run(a.calib_manifest, a.board, a.camera, a.lights, a.out, a.ball_radius_m, a.camera_id,
              noise=a.noise, model_error=a.model_error)
    print(json.dumps({k: rep[k] for k in ("fabric", "fabric_parallel", "sphere") if k in rep}, indent=1))
