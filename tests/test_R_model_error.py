"""R2. Real data is never exactly Lambertian and a real sensor is quiet: with a sheen fabric
and a 14-bit-like noise model, the old absolute-chi^2 confidence drops most of the garment,
the defaults (relative confidence, model-error term 0.005) keep it and keep the lights."""
import numpy as np
import pytest

import gwps.ps as ps
import stage3_mesh_maps
import stage4_ps
from gwps.compare import angle_deg
from gwps.synth import make_dataset

QUIET = (1.5e-4, 60000.0)


@pytest.fixture(scope="module")
def sheen(tmp_path_factory):
    d = tmp_path_factory.mktemp("sheen")
    make_dataset(d / "ds", "torso", None, noise=True, seed=5, angles=[0], brdf="sheen", noise_params=QUIET)
    stage3_mesh_maps.run(d / "ds/sparse/0", d / "ds/mesh.ply", d / "s3", quiet=True)
    return d


def _run(d, model_error, relative):
    orig = ps.local_confidence
    ps.local_confidence = lambda *a, **k: orig(*a, **{**k, "relative": relative})
    try:
        stage4_ps.run(d / "ds/sparse/0", d / "ds/manifest.csv", d / "ds/lights.json", d / "s3", d / "s4",
                      read_noise=QUIET[0], full_well=QUIET[1], model_error=model_error)
    finally:
        ps.local_confidence = orig
    g, p = np.load(d / "ds/gt/step0000.npz"), np.load(d / "s4/step0000.npz")
    m = g["hit"]
    sel = m & p["ok"] & g["unshadowed"]
    return float(np.mean(p["conf"][m] >= 0.3)), float(np.mean(p["n_lights"][m])), \
        float(np.median(angle_deg(p["n_ps"][sel], g["n_gt_cam"][sel])))


def test_confidence_survives_real_data(sheen):
    old = _run(sheen, 0.0, relative=False)
    new = _run(sheen, ps.MODEL_ERROR, relative=True)
    print(f"\n[R2] old (absolute chi^2, no model error): usable {100 * old[0]:.1f} %, lights {old[1]:.2f}, PS median {old[2]:.2f} deg")
    print(f"[R2] defaults: usable {100 * new[0]:.1f} %, lights {new[1]:.2f}, PS median {new[2]:.2f} deg")
    assert old[0] < 0.6
    assert new[0] > 0.95 and new[1] > 7.0 and new[2] < 1.8
