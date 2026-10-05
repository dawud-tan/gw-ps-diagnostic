"""U. Lit-image undistortion (distorted COLMAP camera -> PINHOLE, bilinear, like COLMAP)."""
import csv

import cv2
import numpy as np
import pycolmap
import pytest

import undistort_lit
from gwps.calib_synth import BOARD, DIST_PARAMS, PILOT_CAM, board_albedo_texture, board_pose, render_board
from gwps.camera import Camera, OpenCVCamera
from gwps.charuco import detect
from gwps.undistort import colmap_undistorted_camera, undistort_image, undistort_maps

DCAM = OpenCVCamera(1, PILOT_CAM.width, PILOT_CAM.height, DIST_PARAMS)


def test_identity_without_distortion():
    img = np.random.default_rng(0).uniform(0, 1, (480, 640)).astype(np.float32)
    src = OpenCVCamera.from_colmap(1, "PINHOLE", 640, 480, [800, 801, 330.7, 236.2])
    dst = Camera(1, "PINHOLE", 640, 480, 800, 801, 330.7, 236.2)
    assert np.abs(undistort_image(img, undistort_maps(src, dst)) - img).max() < 1e-4


@pytest.mark.parametrize("model,params", [("SIMPLE_RADIAL", [1100, 640.3, 480.7, -0.08]),
                                          ("OPENCV", [1100, 1101, 652.3, 470.7, -0.08, 0.02, 3e-4, -2e-4])])
def test_matches_colmap(model, params):
    rng = np.random.default_rng(1)
    H, W = 960, 1280
    base = cv2.GaussianBlur(rng.uniform(0, 255, (H // 8, W // 8)).astype(np.float32), (0, 0), 1.5)
    img8 = np.clip(cv2.resize(base, (W, H), interpolation=cv2.INTER_CUBIC), 0, 255).astype(np.uint8)
    src = OpenCVCamera.from_colmap(1, model, W, H, params)
    bm, ucam = pycolmap.undistort_image(pycolmap.UndistortCameraOptions(), pycolmap.Bitmap.from_array(img8),
                                        src.to_pycolmap())
    dst = colmap_undistorted_camera(src)
    assert (dst.width, dst.height) == (ucam.width, ucam.height) and np.allclose(dst.colmap_params(), ucam.params)
    ref = bm.to_array().astype(np.float32)
    ref = ref[..., 0] if ref.ndim == 3 else ref
    d = np.abs(undistort_image(img8.astype(np.float32), undistort_maps(src, dst)) - ref)[5:-5, 5:-5]
    print(f"\n[U] {model}: vs COLMAP mean |diff| {d.mean():.3f}, p99.9 {np.percentile(d, 99.9):.2f} grey levels")
    assert d.mean() < 0.5 and np.percentile(d, 99.9) < 2.0


def _naive_maps(src, dst):
    """Same maps without the COLMAP -> OpenCV half-pixel shift (what a careless call does)."""
    return cv2.initUndistortRectifyMap(src.K, src.dist, None, dst.K, (dst.width, dst.height), cv2.CV_32FC1)


@pytest.mark.parametrize("scale", [1.0, 0.7])
def test_corners_land_on_pinhole_projection(scale):
    """Render the board through the distorted lens, undistort, detect: corners must sit on the
    true projections through the target PINHOLE camera (mean offset within 3 standard errors,
    at most 0.03 px). The 0.7x-focal target makes a half-pixel convention error visible
    (0.5 * (1 - 0.7) = 0.15 px); with COLMAP's own target the focal is kept and it cancels."""
    tex, ppm = board_albedo_texture()
    auto = colmap_undistorted_camera(DCAM)
    dst = Camera(1, "PINHOLE", auto.width, auto.height, auto.fx * scale, auto.fy * scale, auto.cx, auto.cy)
    res = {"ours": [], "naive": []}
    for tilt_y, tilt_x, h in ((25, 20, 0.0), (-30, 20, -0.15), (10, -25, 0.15)):
        R, t = board_pose(tilt_y, tilt_x, h)
        img = render_board(DCAM, R, t, tex, ppm, [], np.random.default_rng(0), noise=False, uniform=0.6)
        for tag, maps in (("ours", undistort_maps(DCAM, dst)), ("naive", _naive_maps(DCAM, dst))):
            det = detect(undistort_image(img, maps), BOARD)
            res[tag].append(det.uv - dst.project(BOARD.corners()[det.ids] @ R.T + t))
    r, rn = np.concatenate(res["ours"]), np.concatenate(res["naive"])
    rms = np.sqrt((r ** 2).sum(1).mean())
    se = r.std(0) / np.sqrt(len(r))
    print(f"\n[U] target focal x{scale}: ours mean signed {np.round(r.mean(0), 3)} px (se {np.round(se, 3)}), "
          f"rms {rms:.3f} px, n={len(r)}; without the half-pixel shift {np.round(rn.mean(0), 3)} px")
    assert np.all(np.abs(r.mean(0)) < np.minimum(3 * se + 0.005, 0.03))
    assert rms < 0.15
    if scale != 1.0:
        assert np.abs(rn.mean(0)).max() > 0.08      # the test can see a convention error


def test_saturation_is_propagated():
    img = np.full((480, 640), 0.3, np.float32)
    yy, xx = np.mgrid[0:480, 0:640]
    sat = (xx - 400) ** 2 + (yy - 200) ** 2 < 15 ** 2
    img[sat] = 1.0
    src = OpenCVCamera.from_colmap(1, "SIMPLE_RADIAL", 640, 480, [700, 320, 240, -0.1])
    maps = undistort_maps(src, colmap_undistorted_camera(src))
    out = undistort_image(img, maps, sat)
    plain = undistort_image(img, maps)
    flagged = out >= 0.98
    mixed = (plain > 0.31) & (plain < 0.98)            # partly saturated after interpolation
    print(f"\n[U] saturated source px {sat.sum()}, flagged after undistortion {flagged.sum()}, "
          f"mixed pixels that would have escaped {mixed.sum()}")
    assert mixed.any() and flagged[mixed].all()


def test_cli_rewrites_manifest(tmp_path):
    rng = np.random.default_rng(2)
    rows = []
    for k in range(2):
        p = tmp_path / "raw" / f"s{k}_L1.png"
        p.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(p), (rng.uniform(0, 0.9, (480, 640)) * 65535).astype(np.uint16))
        rows.append({"step": k, "camera_id": 1, "colmap_image_name": f"s{k}.png", "light_id": 1, "path": f"raw/s{k}_L1.png"})
    with open(tmp_path / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (tmp_path / "cameras.txt").write_text("1 SIMPLE_RADIAL 640 480 700 320 240 -0.05\n")
    man, cams = undistort_lit.run(tmp_path / "manifest.csv", tmp_path / "cameras.txt", "auto", tmp_path / "und", workers=1)
    new = list(csv.DictReader(open(man)))
    assert [{k: v for k, v in r.items() if k != "path"} for r in new] == \
           [{k: str(v) for k, v in r.items() if k != "path"} for r in rows]
    assert all((man.parent / r["path"]).exists() for r in new)
    assert "PINHOLE" in cams.read_text()
