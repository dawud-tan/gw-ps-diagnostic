"""Shared synthetic datasets and a pipeline runner. Datasets are built once per session."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
# pytest.ini's --basetemp=runs/_pytest needs runs/ to exist, and a fresh clone has none (it is
# git-ignored): pytest creates the basetemp itself but not its parent, and every test errors.
(Path(__file__).resolve().parents[1] / "runs").mkdir(exist_ok=True)

import stage3_mesh_maps  # noqa: E402
import stage4_ps  # noqa: E402
import stage5_compare  # noqa: E402
import stage6_floor  # noqa: E402
import stage7_verdict  # noqa: E402
from gwps.synth import make_dataset  # noqa: E402

NOISE = {"none": False, "noisy": True}


class Synth:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}

    def dataset(self, kind="torso", variant=None, noise="none", steps=12):
        """steps: turntable steps over 360 deg for the torso (12 = the original 30 deg steps;
        36 = the garment plan's 10 deg steps)."""
        key = ("ds", kind, variant, noise, steps)
        if key not in self.cache:
            d = self.root / f"{kind}_{variant}_{noise}" if steps == 12 else self.root / f"{kind}_{variant}_{noise}_{steps}steps"
            if not (d / "scene.json").exists():
                angles = None if kind != "torso" or steps == 12 else [360.0 * k / steps for k in range(steps)]
                make_dataset(d, kind, variant, NOISE[noise], seed=hash(key) % 1000, angles=angles)
            self.cache[key] = d
        return self.cache[key]

    def stages34(self, ds, directional=False):
        key = ("s34", str(ds), directional)
        if key not in self.cache:
            run = ds / ("run_dir" if directional else "run")
            stage3_mesh_maps.run(ds / "sparse/0", ds / "mesh.ply", run / "stage3", quiet=True)
            stage4_ps.run(ds / "sparse/0", ds / "manifest.csv", ds / "lights.json", run / "stage3",
                          run / "stage4", directional=directional)
            self.cache[key] = run
        return self.cache[key]

    def stage5(self, ds):
        key = ("s5", str(ds))
        if key not in self.cache:
            run = self.stages34(ds)
            res, summary = stage5_compare.run(ds / "sparse/0", ds / "mesh.ply", ds / "garment_faces.npy",
                                              run / "stage3", run / "stage4", run / "stage5")
            self.cache[key] = (run, res, summary)
        return self.cache[key]

    def floors(self, noise):
        key = ("floor", noise)
        if key not in self.cache:
            b = self.dataset("board", None, noise)
            stage6_floor.run(b / "sparse/0", b / "mesh.ply", b / "garment_faces.npy", b / "manifest.csv",
                             b / "lights.json", b / "run", "fabric")
            self.cache[key] = json.loads((b / "run" / "floors.json").read_text())
        return self.cache[key]

    def verdict(self, variant, noise, steps=12):
        ds = self.dataset("torso", variant, noise, steps)
        run, _, _ = self.stage5(ds)
        res = stage7_verdict.run(run / "stage5", ds / "mesh.ply", ds / "garment_faces.npy",
                                 self.floors(noise), run / "verdict.md")
        return ds, run, res


@pytest.fixture(scope="session")
def synth(tmp_path_factory):
    return Synth(tmp_path_factory.mktemp("synth"))


def fmt(d):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}


from pilot_fixtures import pilot  # noqa: E402,F401  (session fixture for the stage C tests)
