"""Synthetic pilot capture shared by the stage C / board / control tests (built once)."""
import json
from pathlib import Path

import numpy as np
import pytest

import stage6_controls
import stageC_intrinsics
import stageC_lights
from gwps.calib_synth import BALL_R, make_pilot


class Pilot:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}

    def data(self, swap_mm=0.02, positions=6, parts=("balls", "board", "intrinsics", "radiometry"), distorted=False):
        key = ("data", swap_mm, positions, parts, distorted)
        if key not in self.cache:
            d = self.root / f"pilot_swap{swap_mm}_n{positions}_{'-'.join(parts)}{'_dist' if distorted else ''}"
            if not (d / "truth.json").exists():
                make_pilot(d, positions, swap_mm / 1000, True, seed=11, parts=parts, distorted=distorted)
            self.cache[key] = d
        return self.cache[key]

    def truth(self, d):
        return json.loads((d / "truth.json").read_text())

    def lights(self, d, source="mirror"):
        key = ("lights", str(d), source)
        if key not in self.cache:
            out = d / f"run_{source}"
            _, rep = stageC_lights.run(d / "calib_manifest.csv", d / "pinhole/cameras.txt", BALL_R,
                                       out / "lights.json", out / "stageC", position_source=source)
            self.cache[key] = (out / "lights.json", rep)
        return self.cache[key]

    def intrinsics(self, d):
        key = ("intr", str(d))
        if key not in self.cache:
            self.cache[key] = stageC_intrinsics.run(d / "calib_manifest.csv", d / "board.json", d / "intrinsics.json")
        return self.cache[key]

    def controls(self, d, calibrated=True):
        key = ("controls", str(d), calibrated)
        if key not in self.cache:
            if calibrated:
                lights, _ = self.lights(d)
                rep = stage6_controls.run(d / "calib_manifest.csv", d / "board.json", d / "pinhole/cameras.txt",
                                          lights, d / "controls_cal", BALL_R)
            else:
                t = self.truth(d)
                poses = {k: (np.array(v["R"]), np.array(v["t"])) for k, v in t["board_poses"].items()
                         if not k.startswith("i")}
                cen = {k: np.array(v) for k, v in t["matte_centres_m"].items()}
                rep = stage6_controls.run(d / "calib_manifest.csv", d / "board.json", d / "pinhole/cameras.txt",
                                          d / "lights_true.json", d / "controls_true", BALL_R,
                                          poses_override=poses, centres_override=cen)
            self.cache[key] = rep
        return self.cache[key]


@pytest.fixture(scope="session")
def pilot(tmp_path_factory):
    return Pilot(tmp_path_factory.mktemp("pilot"))
