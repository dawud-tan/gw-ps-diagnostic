"""C7. The real-data flow on a pilot rendered through a distorted lens:
intrinsics (ChArUco) -> stage C on raw images -> undistort_lit -> stage 6 floors."""
import json

import numpy as np

import stage6_controls
import stageC_intrinsics
import stageC_lights
import undistort_lit
from gwps.calib_synth import BALL_R


def test_distorted_pilot_flow(pilot):
    d = pilot.data(0.02, 6, distorted=True)
    run = d / "flow"
    intr = run / "intrinsics.json"
    stageC_intrinsics.run(d / "calib_manifest.csv", d / "board.json", intr)
    out = {}
    for src in ("mirror", "shading"):
        _, rep = stageC_lights.run(d / "calib_manifest.csv", intr, BALL_R, run / f"lights_{src}.json",
                                   run / f"stageC_{src}", position_source=src)
        t = pilot.truth(d)
        out[src] = np.array([1000 * np.linalg.norm(np.array(r["position_used_m"]) - np.array(t["lights"][str(k)]["position_m"]))
                             for k, r in rep["lights"].items()])
    man, cams = undistort_lit.run(d / "calib_manifest.csv", intr, "auto", run / "undist")
    floors = stage6_controls.run(man, d / "board.json", cams, run / "lights_mirror.json", run / "controls", BALL_R)
    print(f"\n[C7] light errors (mm), mirror: {np.round(out['mirror'], 1)}, shading refit: {np.round(out['shading'], 1)}")
    for kind in ("fabric", "sphere"):
        print(f"[C7] {kind} floor: " + ", ".join(f"s={s}: {v:.3f}" for s, v in floors[kind].items()) + " deg")
    assert out["shading"].max() <= 5.0
    for s in (5, 10, 20, 50):
        assert floors["fabric"][s] < 0.25
