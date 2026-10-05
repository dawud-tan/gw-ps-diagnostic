"""R1. Radiometry from the pilot's frames: noise model (photon transfer), linearity with a
saturation shoulder, additivity of lights, and LED drift; the drift gate catches 3 %."""
import json

import pytest

import stageC_radiometry
from gwps.calib_synth import make_pilot
from gwps.synth import FULL_WELL, READ_NOISE


@pytest.fixture(scope="module")
def radiometry_runs(tmp_path_factory):
    out = {}
    for tag, drift in (("ok", 0.003), ("drifting", 0.03)):
        d = tmp_path_factory.mktemp(f"radiometry_{tag}")
        make_pilot(d, 6, 20e-6, True, seed=3, parts=("balls", "radiometry"), drift=drift)
        out[tag] = stageC_radiometry.run(d / "calib_manifest.csv", d / "radiometry.json")
    return out


def test_noise_model_and_gates(radiometry_runs):
    r = radiometry_runs["ok"]
    n, c = r["noise_fit"], r["checks"]
    print(f"\n[R1] read {n['read']:.5f} (true {READ_NOISE}), full well {n['full_well']:.0f} (true {FULL_WELL:.0f}), "
          f"from {n['read_from']}; linearity R^2 {c['linearity']['r2']:.7f}, dev {c['linearity']['max_rel_dev_linear_range']:.4f}; "
          f"additivity {json.dumps(c['additivity'])}; drift {c['drift']['ratio']:.4f}")
    assert abs(n["read"] / READ_NOISE - 1) < 0.05
    assert abs(n["full_well"] / FULL_WELL - 1) < 0.05
    assert r["pass"]
    assert abs(c["drift"]["ratio"] - 1.003) < 0.001


def test_drift_gate_catches_3_percent(radiometry_runs):
    r = radiometry_runs["drifting"]
    print(f"\n[R1] drifting lights: ratio {r['checks']['drift']['ratio']:.4f}, pass {r['pass']}")
    assert not r["checks"]["drift"]["pass"] and not r["pass"]
