"""A. Albedo pipeline: colour correction from a ColorChecker in the ChArUco board window
(synthetic camera with channel mixing and gains), applied to a colour garment."""
import json

import cv2
import numpy as np
import pytest

import stage3_mesh_maps
import stage4_ps
import stage_albedo
from gwps import albedo as alb
from gwps.calib_synth import make_pilot
from gwps.synth import make_dataset


@pytest.fixture(scope="module")
def chart(tmp_path_factory):
    d = tmp_path_factory.mktemp("albedo")
    P = make_pilot(d / "pilot", parts=("colour",), seed=2)
    t = json.loads((P / "truth.json").read_text())
    (P / "reference.json").write_text(json.dumps(t["chart_reference_rendered"]))
    r = stage_albedo.calibrate(P / "calib_manifest.csv", P / "board.json", P / "chart.json", P / "pinhole/cameras.txt",
                               P / "lights_true.json", d / "ccm.json", reference=P / "reference.json")
    return d, r, np.array(t["camera_mix"])


def test_colour_correction_from_chart(chart):
    d, r, mix = chart
    lin = r["fits"]["linear"]
    print(f"\n[A] chart: linear LOO dE00 median {lin['loo_de00_median']:.2f} max {lin['loo_de00_max']:.2f}; "
          f"root-poly median {r['fits']['rootpoly']['loo_de00_median']:.2f}; chosen {r['method']}")
    assert lin["loo_de00_median"] < 1.0 and r["fits"]["rootpoly"]["loo_de00_median"] < 1.0
    assert np.allclose(np.array(lin["matrix"]) @ mix, np.eye(3), atol=0.02)      # undoes the camera's mixing


@pytest.mark.parametrize("noise", [False, True])
def test_garment_albedo(chart, noise, tmp_path):
    d, _, _ = chart
    ds = tmp_path / "torso"
    make_dataset(ds, "torso", None, noise=noise, seed=6, angles=[0, 45], rgb=True)
    stage3_mesh_maps.run(ds / "sparse/0", ds / "mesh.ply", tmp_path / "s3", quiet=True)
    stage4_ps.run(ds / "sparse/0", ds / "manifest.csv", ds / "lights.json", tmp_path / "s3", tmp_path / "s4")
    stage_albedo.apply(ds / "sparse/0", ds / "manifest.csv", ds / "lights.json", tmp_path / "s3", tmp_path / "s4",
                       d / "ccm.json", tmp_path / "alb")
    g, a = np.load(ds / "gt/step0000.npz"), np.load(tmp_path / "alb/step0000.npz")
    sel = a["valid"] & g["unshadowed"]
    de = alb.delta_e00(a["albedo_rgb"][sel], g["albedo_rgb"][sel])
    m = cv2.erode(sel.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    de5 = alb.delta_e00(cv2.blur(a["albedo_rgb"], (5, 5))[m], cv2.blur(g["albedo_rgb"], (5, 5))[m])
    print(f"\n[A] garment albedo ({'noisy' if noise else 'noise-free'}): per-pixel dE00 median {np.median(de):.2f}, "
          f"5x5-averaged median {np.median(de5):.2f}")
    assert np.median(de5) < (0.6 if noise else 0.3)
    assert (tmp_path / "alb/step0000_albedo_srgb.png").exists()
