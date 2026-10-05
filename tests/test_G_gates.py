"""G. Stage 1 (principal-point gate and identical crop) and stage 2 (frame gates), each with
an injected failure the gate must catch."""
import shutil

import cv2
import numpy as np
import pytest
import trimesh

import stage1_gw_prep
import stage2_frame_gate
from gwps.camera import read_cameras_txt, read_images_txt, write_colmap_txt, read_points3D_txt_full
from gwps.io import Manifest
from gwps.synth import make_dataset


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    d = tmp_path_factory.mktemp("gates")
    make_dataset(d / "ds", "torso", None, noise=True, seed=4, angles=[0, 30])
    rep = stage1_gw_prep.run(d / "ds", d / "ds/manifest.csv", d / "gw")
    return d, rep


def _copy(prepared, tmp_path):
    d, _ = prepared
    shutil.copytree(d / "gw", tmp_path / "gw")
    return d, tmp_path / "gw"


def test_stage1_crops_to_a_centred_principal_point(prepared):
    d, rep = prepared
    g = rep["gates"]["1"]
    cam = read_cameras_txt(d / "gw/sparse/0/cameras.txt")[1]
    print(f"\n[G] principal point off by ({g['dx_px']:.1f}, {g['dy_px']:.1f}) px -> crop {g['crop']} -> "
          f"residual ({cam.cx - cam.width / 2:.2f}, {cam.cy - cam.height / 2:.2f}) px")
    assert not g["pass"] and abs(cam.cx - cam.width / 2) < 0.5 and abs(cam.cy - cam.height / 2) < 0.5
    x0, y0, w, h = g["crop"]
    orig = Manifest(d / "ds/manifest.csv").views["step0000.png"]["lights"][3]
    new = Manifest(d / "gw/manifest.csv").views["step0000.png"]["lights"][3]
    assert np.array_equal(np.load(new), np.load(orig)[y0:y0 + h, x0:x0 + w])        # identical crop of lit images
    assert cv2.imread(str(d / "gw/images/step0000.png")).shape[:2] == (h, w)


def test_stage1_leaves_a_centred_camera_alone(prepared, tmp_path):
    d, _ = prepared
    shutil.copytree(d / "ds", tmp_path / "ds")
    cams = read_cameras_txt(tmp_path / "ds/sparse/0/cameras.txt")
    c = cams[1]
    c.cx, c.cy = c.width / 2, c.height / 2
    write_colmap_txt(tmp_path / "ds/sparse/0", cams, read_images_txt(tmp_path / "ds/sparse/0/images.txt"),
                     read_points3D_txt_full(tmp_path / "ds/sparse/0/points3D.txt"))
    rep = stage1_gw_prep.run(tmp_path / "ds", tmp_path / "ds/manifest.csv", tmp_path / "gw")
    assert rep["gates"]["1"]["pass"] and rep["gates"]["1"]["crop"] is None


def test_stage2_passes_on_clean_data(prepared, tmp_path):
    d, _ = prepared
    r = stage2_frame_gate.run(d / "gw", d / "ds/mesh.ply", tmp_path / "s2", manifest=d / "gw/manifest.csv")
    print("\n" + (tmp_path / "s2/stage2.md").read_text())
    assert r["pass"] and (tmp_path / "s2/overlays/step0000.png").exists()


def test_stage2_catches_half_pixel_observations(prepared, tmp_path):
    d, gw = _copy(prepared, tmp_path)
    ims = read_images_txt(gw / "sparse/0/images.txt")
    for im in ims.values():
        im.xys = im.xys + np.array([0.5, 0.0])
    write_colmap_txt(gw / "sparse/0", read_cameras_txt(gw / "sparse/0/cameras.txt"), ims,
                     read_points3D_txt_full(gw / "sparse/0/points3D.txt"))
    r = stage2_frame_gate.run(gw, d / "ds/mesh.ply", tmp_path / "s2", manifest=gw / "manifest.csv")
    assert not r["pass"] and not r["views"]["step0000.png"]["reprojection"]["pass"]


def test_stage2_catches_wrong_mesh_scale(prepared, tmp_path):
    d, _ = prepared
    m = trimesh.load(d / "ds/mesh.ply", process=False)
    m.apply_scale(1.1)
    m.export(tmp_path / "scaled.ply")
    r = stage2_frame_gate.run(d / "gw", tmp_path / "scaled.ply", tmp_path / "s2", manifest=d / "gw/manifest.csv")
    print(f"\n[G] mesh x1.1: depth median |dz| {1000 * r['views']['step0000.png']['depth']['median_abs_m']:.1f} mm")
    assert not r["views"]["step0000.png"]["depth"]["pass"]


def _shift_npy(p, dx, dy):
    x = np.load(p)
    np.save(p, cv2.warpAffine(x, np.float32([[1, 0, dx], [0, 1, dy]]), x.shape[::-1], flags=cv2.INTER_CUBIC))


def test_stage2_catches_lit_set_offset_and_bumped_frame(prepared, tmp_path):
    d, gw = _copy(prepared, tmp_path)
    man = Manifest(gw / "manifest.csv")
    v0, v1 = man.views["step0000.png"], man.views["step0001.png"]
    for p in list(v0["lights"].values()) + [v0["ambient"]]:
        _shift_npy(p, 0.3, 0.0)                      # the whole lit set of view 0 is offset
    _shift_npy(v1["lights"][4], 1.0, 0.0)            # one bumped frame in view 1
    r = stage2_frame_gate.run(gw, d / "ds/mesh.ply", tmp_path / "s2", manifest=gw / "manifest.csv")
    a0, a1 = r["views"]["step0000.png"]["alignment"], r["views"]["step0001.png"]["alignment"]
    print(f"\n[G] set offset 0.3 px -> measured {np.round(a0['set_shift_px'], 3)}; light 4 -> "
          f"{np.round(a1['per_light_px']['4'], 3)} (bumped 1.0 px), failing {a1['failing_lights']}")
    assert not a0["pass"] and not a1["pass"] and a1["failing_lights"] == ["4"]
