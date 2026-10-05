"""S. The COLMAP chain on a synthetic garment session (36 x 10 deg steps with 1 deg of turntable
jitter, distorted lens with an off-centre principal point, textured static studio):
silhouette masks -> SfM with the pilot's (here true) intrinsics fixed -> turntable frame in metres
from the metric-board frames -> undistorted images and masks -> stage 1 (RGBA, principal point
centred) -> stage 2 with the true mesh. Plus the board-axis fit on its own, a board that slipped,
and the self-calibrated fallback."""
import json
import shutil

import cv2
import numpy as np
import pytest
import trimesh

import stage1_gw_prep
import stage2_frame_gate
import stage_masks
import stage_sfm
from gwps.calib_synth import BOARD, DIST_PARAMS
from gwps.camera import OpenCVCamera, read_cameras_txt, read_images_txt
from gwps.capture import develop_for_colmap
from gwps.charuco import Detection
from gwps.sfm import fit_board_axis
from gwps.synth import Rz
from gwps.synth_sfm import make_session, metric_board_pose, step_poses


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    d = tmp_path_factory.mktemp("sfm")
    s = make_session(d / "session", n_steps=36, jitter_deg=1.0, seed=3)
    develop_for_colmap(s)
    stage_masks.run(s, "silhouette", s / "static_exclude.png")
    rep = stage_sfm.run(s, s / "intrinsics_true.json", s / "board.json", d / "sfm", camera_height_m=0.30,
                        measured_axis_distance_m=1.60)
    gw = stage1_gw_prep.run(d / "sfm/undistorted", d / "sfm/lit/manifest.csv", d / "gw", masks=d / "sfm/undistorted/masks")
    return s, d, rep, gw


def _truth_frame(s):
    """Truth world -> the chain's frame (step 0's camera on -Y; the jitter moved it)."""
    t = json.loads((s / "truth.json").read_text())
    C0 = np.array(t["images"]["step0000.png"]["centre"])
    return t, Rz(-(np.degrees(np.arctan2(C0[1], C0[0])) + 90.0))


def test_sfm_gates(chain):
    s, d, rep, _ = chain
    print("\n" + (d / "sfm/sfm.md").read_text())
    assert rep["pass"] and rep["sfm"]["n_registered"] == 36
    assert rep["sfm"]["mean_reprojection_error_px"] < 0.5
    assert abs(rep["metric_board"]["camera_axis_distance_m"] - 1.60) / 1.60 < 0.002          # scale within 0.2 %
    assert rep["alignment"]["axis_angle_to_board_deg"] < 0.1


def test_poses_match_truth(chain):
    s, d, rep, _ = chain
    t, Q = _truth_frame(s)
    ims = read_images_txt(d / "sfm/aligned/images.txt")
    dc, dr = [], []
    for im in ims.values():
        dc.append(1000 * np.linalg.norm(im.centre - Q @ np.array(t["images"][im.name]["centre"])))
        dR = im.R @ (np.array(t["images"][im.name]["R"]) @ Q.T).T
        dr.append(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    print(f"\n[S] camera centres {np.sqrt(np.mean(np.square(dc))):.3f} mm RMS (max {max(dc):.3f}); "
          f"rotations {np.sqrt(np.mean(np.square(dr))):.4f} deg RMS (max {max(dr):.4f})")
    assert np.sqrt(np.mean(np.square(dc))) < 2.0 and max(dr) < 0.1


def test_stage1_rgba_and_stage2_with_true_mesh(chain, tmp_path):
    s, d, rep, gw = chain
    g = gw["gates"]["1"]
    cam = read_cameras_txt(d / "gw/sparse/0/cameras.txt")[1]
    assert not g["pass"] and abs(cam.cx - cam.width / 2) < 0.5 and abs(cam.cy - cam.height / 2) < 0.5
    img = cv2.imread(str(d / "gw/images/step0003.png"), cv2.IMREAD_UNCHANGED)
    und = cv2.imread(str(d / "sfm/undistorted/masks/step0003.png"), cv2.IMREAD_GRAYSCALE)
    x0, y0, w, h = g["crop"]
    assert img.shape == (h, w, 4) and np.array_equal(img[..., 3] > 127, und[y0:y0 + h, x0:x0 + w] > 127)
    _, Q = _truth_frame(s)
    mesh = trimesh.load(s / "mesh_true.ply", process=False)
    T = np.eye(4)
    T[:3, :3] = Q
    mesh.apply_transform(T)
    mesh.export(tmp_path / "mesh.ply")
    r = stage2_frame_gate.run(d / "gw", tmp_path / "mesh.ply", tmp_path / "s2", manifest=d / "gw/manifest.csv")
    v = list(r["views"].values())
    dz = max(x["depth"]["median_abs_m"] for x in v)
    outside = max(x["mask"]["mesh_hits_outside_mask"] for x in v)
    print(f"\n[S] stage 2 on the chain's output: pass={r['pass']}, depth median |dz| max {1000 * dz:.3f} mm, "
          f"mesh pixels outside the mask max {100 * outside:.3f} %")
    assert r["pass"] and dz < 0.0005 and outside < 0.005


def test_board_axis_fit_recovers_the_axis():
    """Exact synthetic corners (plus 0.1 px noise) of the board at five turntable angles."""
    cam = OpenCVCamera(1, 1920, 1440, DIST_PARAMS)
    rng = np.random.default_rng(0)
    corners = BOARD.corners()
    dets, poses = [], []
    for a in (-40, -20, 0, 20, 40):
        R, t = metric_board_pose(a)
        uv = cam.project(corners @ R.T + t) + rng.normal(0, 0.1, (len(corners), 2))
        dets.append(Detection(np.arange(len(corners)), uv, 0))
        poses.append((R @ cv2.Rodrigues(rng.normal(0, 2e-3, 3))[0], t + rng.normal(0, 1e-3, 3)))   # rough PnP
    fit = fit_board_axis(dets, poses, corners, cam)
    Rc, _ = step_poses([0.0])[0]
    axis_true = Rc @ np.array([0, 0, 1.0])                          # the turntable axis in the camera frame
    ang = np.degrees(np.arccos(min(1.0, abs(float(fit["axis"] @ axis_true)))))
    print(f"\n[S] board axis: {ang:.4f} deg off, camera-to-axis {fit['camera_axis_distance_m']:.5f} m (1.6 true), "
          f"{fit['rms_px']:.3f} px, angles {np.round(fit['angles_deg'], 3)}")
    assert ang < 0.05 and abs(fit["camera_axis_distance_m"] - 1.6) < 0.001
    assert np.allclose(np.abs(np.diff(fit["angles_deg"])), 20, atol=0.05)


def test_a_board_that_slipped_fails_the_gate(chain, tmp_path):
    s, d, _, _ = chain
    s2 = tmp_path / "session"
    shutil.copytree(s / "calib", s2 / "calib")
    shutil.copy(s / "calib_manifest.csv", s2 / "calib_manifest.csv")
    p = s2 / "calib/metric_board/m3/all.png"                       # the board moved ~8 px (~4 mm) before m3
    img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    cv2.imwrite(str(p), cv2.warpAffine(img, np.float32([[1, 0, 8], [0, 1, 0]]), img.shape[::-1]))
    fit = stage_sfm.metric_board_fit(s2 / "calib_manifest.csv", s / "board.json", OpenCVCamera(1, 1920, 1440, DIST_PARAMS))
    print(f"\n[S] slipped board: axis model {fit['rms_px']:.2f} px vs free poses {fit['free_rms_px']:.2f} px, "
          f"corners {fit['corner_rms_mm']:.2f} mm RMS")
    assert fit["corner_rms_mm"] > stage_sfm.GATES["board_axis_rms_mm"]


def test_self_calibration_is_never_silently_wrong(chain, tmp_path):
    """Without the pilot's intrinsics COLMAP self-calibrates with the principal point fixed at the
    image centre (this lens's is 30 px off) and SIMPLE_RADIAL cannot model k2 or the tangential
    terms: the scale came out 2.5 % off. The chain must then fail a gate rather than pass."""
    s, _, _, _ = chain
    rep = stage_sfm.run(s, None, s / "board.json", tmp_path / "sfm", camera_height_m=0.30, lit=False,
                        self_calibration_model="SIMPLE_RADIAL")
    err = abs(rep["metric_board"]["camera_axis_distance_m"] - 1.60) / 1.60
    print(f"\n[S] self-calibrated SIMPLE_RADIAL: registered {rep['sfm']['n_registered']}/36, reprojection "
          f"{rep['sfm']['mean_reprojection_error_px']:.3f} px, scale off {100 * err:.2f} %, pass={rep['pass']} {rep['gates']}")
    assert rep["sfm"]["n_registered"] == 36
    assert err < 0.01 or not rep["pass"]
