"""R3. A 27 MP view (6000 x 4500) through stages 3-5, each in its own process: peak memory
per stage under 6 GB (this machine has 15 GB)."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from gwps.synth import make_dataset

ROOT = Path(__file__).resolve().parents[1]
BUDGET_GB = 6.0


@pytest.mark.slow
def test_27mp_view_memory(tmp_path):
    ds = tmp_path / "ds"
    make_dataset(ds, "torso", None, noise=True, seed=9, angles=[0], resolution_scale=9.375)
    py = sys.executable
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    cmds = {"stage3": [py, ROOT / "src/stage3_mesh_maps.py", "--sparse", ds / "sparse/0", "--mesh", ds / "mesh.ply", "--out", tmp_path / "s3"],
            "stage4": [py, ROOT / "src/stage4_ps.py", "--sparse", ds / "sparse/0", "--manifest", ds / "manifest.csv",
                       "--lights", ds / "lights.json", "--stage3", tmp_path / "s3", "--out", tmp_path / "s4"],
            "stage5": [py, ROOT / "src/stage5_compare.py", "--sparse", ds / "sparse/0", "--mesh", ds / "mesh.ply",
                       "--garment-faces", ds / "garment_faces.npy", "--stage3", tmp_path / "s3", "--stage4", tmp_path / "s4",
                       "--out", tmp_path / "s5"]}
    print("")
    for name, cmd in cmds.items():
        p = subprocess.Popen([str(c) for c in cmd], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        _, status, ru = os.wait4(p.pid, 0)
        gb = ru.ru_maxrss / 1024 ** 2
        print(f"[R3] {name}: exit {os.waitstatus_to_exitcode(status)}, peak RSS {gb:.2f} GB")
        assert os.waitstatus_to_exitcode(status) == 0, p.stderr.read().decode()[-2000:]
        assert gb < BUDGET_GB
