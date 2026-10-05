"""W. The GW dry-run kit. (1) gw_pod.sh run/pack against stand-ins for conda, nvcc, nvidia-smi and GW's
script, so the paid pod does not find its bugs. (2) gw_dryrun.py prepare -> finish on a synthetic
garment session with PS frames and a 3 mm bump, with stand-ins for GW's mesh in stage_sfm's frame:
the true surface without the bump must come out BUILD with the bump flagged, the true surface BAKE.
This is also the whole real-data chain after capture: masks, COLMAP, stage 1, stages 2-7, the bake."""
import json
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import gw_dryrun

ROOT = Path(__file__).resolve().parents[1]


def _exe(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def test_pod_script_run_and_pack(tmp_path):
    work, stub = tmp_path / "work", tmp_path / "stub"
    _exe(work / "miniforge3/bin/conda", "#!/bin/sh\nexit 0\n")
    (work / "miniforge3/etc/profile.d").mkdir(parents=True)
    (work / "miniforge3/etc/profile.d/conda.sh").write_text("conda() { return 0; }\n")
    _exe(tmp_path / "cuda/bin/nvcc", "#!/bin/sh\necho 'Cuda compilation tools, release 12.1, V12.1.105'\n")
    _exe(stub / "nvidia-smi", "#!/bin/sh\nwhile true; do echo '2026/09/29 10:00:00.000, 20480, 49140, 90'; "
                              "echo '2026/09/29 10:00:05.000, 31744, 49140, 97'; sleep 0.2; done\n")
    _exe(stub / "python", f"#!/bin/sh\nexec {sys.executable} \"$@\"\n")
    _exe(work / "GaussianWrapping/gaussian_wrapping/scripts/train_and_extract_gw_radegs.py",
         "import sys, os\na = sys.argv\nm = a[a.index('-m') + 1]\nassert '--no_postprocess' in a and a[a.index('-r') + 1] == '1'\n"
         "open(os.path.join(m, 'mesh_exact_computation_2pivots_searched.ply'), 'w').write('ply')\n"
         "print('[INFO] Step 1/3: Training...')\n")
    ds = tmp_path / "gw_dataset"
    (ds / "sparse/0").mkdir(parents=True)
    (ds / "sparse/0/cameras.txt").write_text("")
    env = {**os.environ, "WORK": str(work), "CUDA_ROOT": str(tmp_path / "cuda"), "PATH": f"{stub}:{os.environ['PATH']}"}
    out = tmp_path / "out"
    r = subprocess.run(["bash", str(ROOT / "src/gw_pod.sh"), "run", str(ds), str(out)], env=env, capture_output=True,
                       text=True, timeout=120)
    print(r.stdout[-600:], r.stderr[-600:])
    assert r.returncode == 0 and "peak VRAM 31.0 GB" in r.stdout
    assert (out / "mesh_exact_computation_2pivots_searched.ply").exists() and "Step 1/3" in (out / "gw.log").read_text()
    st = gw_dryrun.pod_stats(out)
    assert st["peak_vram_gb"] == pytest.approx(31.0) and st["wall_h"] >= 0 and st["peak_host_ram_gb"] > 0
    r = subprocess.run(["bash", str(ROOT / "src/gw_pod.sh"), "pack", str(out)], env=env, capture_output=True, text=True)
    assert r.returncode == 0
    with tarfile.open(tmp_path / "out.tgz") as t:
        names = set(t.getnames())
    assert {"out/mesh_exact_computation_2pivots_searched.ply", "out/gpu.csv", "out/time.txt", "out/gw.log"} <= names


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    work = tmp_path_factory.mktemp("gw_dryrun")
    rep = gw_dryrun.prepare(work, steps=12, scale=1.0, bump=True)
    return work, rep


def _stand_in(work, bump, path):
    m, _ = gw_dryrun.truth_mesh_in_chain_frame(work / "session", bump)
    m.export(path)
    return path


def test_prepare(prepared):
    work, rep = prepared
    print(f"\n[W] prepare: stage_sfm {rep['sfm_gates']}; upload {rep['upload_mb']:.0f} MB; camera {rep['cameras']['1']}")
    assert rep["sfm_pass"] and (work / "gw_dataset_upload.tar").exists()
    assert (work / "board_run/floors.json").exists()


def test_finish_flags_a_mesh_that_lost_the_bump(prepared, tmp_path):
    work, _ = prepared
    rep = gw_dryrun.finish(work, _stand_in(work, False, tmp_path / "smooth.ply"), out=tmp_path / "finish")
    print("\n" + (tmp_path / "finish/finish.md").read_text())
    assert rep["stage2_pass"] and rep["verdict"] == "BUILD candidate" and rep["bump_found"]
    assert 2.5 < rep["to_true_surface_mm"]["max"] < 3.5 and rep["to_true_surface_mm"]["median"] < 0.05   # the 3 mm bump
    h = [c["height_fine_mm"] for sc in ("5", "10") for c in rep["clusters"].get(sc, []) if c["height_fine_mm"]]
    assert h and 2.0 < max(h) < 3.2                                    # the clusters say how deep: ~3 mm missing
    assert rep["bake"]["faces_decimated"] <= 50_000


def test_finish_passes_the_true_surface(prepared, tmp_path):
    work, _ = prepared
    rep = gw_dryrun.finish(work, _stand_in(work, True, tmp_path / "true.ply"), out=tmp_path / "finish", bake=False)
    print("\n" + (tmp_path / "finish/finish.md").read_text())
    assert rep["stage2_pass"] and rep["verdict"] == "BAKE" and rep["to_true_surface_mm"]["max"] < 0.2
    json.dumps(rep)                                            # the report is plain JSON
