"""Benchmark stages 3-7 and the bake at GW-like mesh sizes, on the synthetic torso with the bump.

  bench_mesh_size.py --faces 10e6 --scale 9.375 --views 4 --out runs/mesh_size_10M

Renders a torso dataset (noisy) whose mesh has ~--faces faces, seen by --scale times the 640 x 480
synthetic camera (9.375 gives 6000 x 4500, 24 MP) at 30 deg steps, and a flat-board control at the
same scale; then runs stages 3, 4, 5, the fabric floor (stage 6), 7 and the bake, each in its own
process under /usr/bin/time. Writes bench.json and bench.md: wall time and peak memory per stage,
the verdict, and whether the bump was found.

Faces smaller than a pixel's footprint mostly go unseen (a face gets values only from pixel centres
that land on it): at 1.5 m that footprint is ~0.25 mm^2 at scale 3 (a torso of ~2M faces) and
~0.026 mm^2 at 24 MP (~20M faces).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

SRC = Path(__file__).resolve().parent


def tessellation(faces):
    """(n_theta, n_z) of gwps.synth.torso_mesh giving ~faces faces, in its default proportions."""
    n_theta = int(round(np.sqrt(faces / (2 * 200 / 340))))
    return n_theta, int(round(faces / (2 * n_theta))) + 1


def timed(name, args, log):
    """Run a stage script in its own process: -> (seconds, peak RSS in GB)."""
    cmd = ["/usr/bin/time", "-f", "%e %M", sys.executable, str(SRC / f"{name}.py")] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(SRC)})
    log.write_text(r.stdout + r.stderr)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"{name} failed ({r.returncode}); see {log}")
    s, kb = r.stderr.strip().splitlines()[-1].split()
    return float(s), int(kb) / 1024 ** 2


def render(out, scale, views, tess, noise):
    """The torso (bump) and board datasets, rendered in a child process so their memory is freed
    before the stages run (at 24 MP the renderer's heap stays at ~5 GB)."""
    code = ("import sys; from gwps.synth import make_dataset; "
            f"make_dataset({str(out / 'torso')!r}, 'torso', 'bump', {noise}, seed=3, "
            f"angles=[-30.0 + 30.0 * k for k in range({views})], resolution_scale={scale}, tessellation={tuple(tess)}); "
            f"make_dataset({str(out / 'board')!r}, 'board', None, {noise}, seed=4, resolution_scale={scale})")
    subprocess.run([sys.executable, "-c", code], check=True, env={**os.environ, "PYTHONPATH": str(SRC)})
    return out / "torso", out / "board"


def run(out, faces=2e6, scale=3.0, views=4, noise=True, target_faces=50_000):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tess = tessellation(faces)
    rep = {"faces_target": faces, "tessellation": tess, "scale": scale, "views": views, "stages": {}}
    t0 = time.time()
    ds, board = render(out, scale, views, tess, noise)
    rep["render_s"] = time.time() - t0
    mesh = trimesh.load(ds / "mesh.ply", process=False)
    rep["faces"] = int(len(mesh.faces))
    info = json.loads((ds / "scene.json").read_text())
    run_dir, logs = out / "run", out / "logs"
    logs.mkdir(exist_ok=True)
    steps = [
        ("stage3_mesh_maps", ["--sparse", ds / "sparse/0", "--mesh", ds / "mesh.ply", "--out", run_dir / "stage3"]),
        ("stage4_ps", ["--sparse", ds / "sparse/0", "--manifest", ds / "manifest.csv", "--lights", ds / "lights.json",
                       "--stage3", run_dir / "stage3", "--out", run_dir / "stage4"]),
        ("stage5_compare", ["--sparse", ds / "sparse/0", "--mesh", ds / "mesh.ply", "--garment-faces", ds / "garment_faces.npy",
                            "--stage3", run_dir / "stage3", "--stage4", run_dir / "stage4", "--out", run_dir / "stage5"]),
        ("stage6_floor", ["--sparse", board / "sparse/0", "--mesh", board / "mesh.ply", "--garment-faces",
                          board / "garment_faces.npy", "--manifest", board / "manifest.csv", "--lights", board / "lights.json",
                          "--run-dir", out / "board_run"]),
        ("stage7_verdict", ["--stage5", run_dir / "stage5", "--mesh", ds / "mesh.ply", "--garment-faces",
                            ds / "garment_faces.npy", "--floors", out / "board_run/floors.json", "--out", run_dir / "verdict.md"]),
        ("stage_bake", ["--sparse", ds / "sparse/0", "--mesh", ds / "mesh.ply", "--garment-faces", ds / "garment_faces.npy",
                        "--stage3", run_dir / "stage3", "--stage4", run_dir / "stage4", "--out", run_dir / "bake/garment",
                        "--target-faces", target_faces]),
    ]
    for name, args in steps:
        s, gb = timed(name, args, logs / f"{name}.log")
        rep["stages"][name] = {"seconds": s, "peak_rss_gb": gb}
    v = json.loads((run_dir / "verdict.json").read_text())
    c = np.array(info["bump_centre_world_m"])
    near = np.linalg.norm(mesh.triangles_center - c, axis=1) < 2 * info["bump_sigma_m"]
    res = dict(np.load(run_dir / "stage5/faces.npz"))
    seen = res["n_views_any"] > 0
    rep["faces_seen_fraction"] = float(seen[np.load(ds / "garment_faces.npy")].mean())
    bump = []
    for s in ("5", "10"):
        for cl in v["clusters"][s]:
            if np.linalg.norm(np.array(cl["centroid_world_m"]) - c) < 0.03:
                bump.append({"scale_mm": int(s), "area_mm2": cl["area_mm2"], "mrl": cl["mrl"]})
    rep["verdict"] = v["verdict"]
    rep["bump_clusters"] = bump
    rep["faces_near_bump"] = int(near.sum())
    rep["bake"] = json.loads((run_dir / "bake/garment.json").read_text())
    (out / "bench.json").write_text(json.dumps(rep, indent=1, default=float))
    L = [f"# Mesh-size benchmark: {rep['faces']:,} faces, {views} views at {int(640 * scale)} x {int(480 * scale)}", "",
         f"Rendering {rep['render_s']:.0f} s. Garment faces seen by some view: {100 * rep['faces_seen_fraction']:.1f} %.",
         f"Verdict: {v['verdict']}; bump clusters: {bump}. Bake: {rep['bake']['faces_garment']:,} -> "
         f"{rep['bake']['faces_decimated']:,} faces, coverage {rep['bake']['texel_coverage_by_views']:.4f}.", "",
         "| stage | wall (s) | peak memory (GB) |", "|---|---|---|"]
    L += [f"| {k} | {x['seconds']:.1f} | {x['peak_rss_gb']:.2f} |" for k, x in rep["stages"].items()]
    (out / "bench.md").write_text("\n".join(L) + "\n")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--faces", type=float, default=2e6)
    ap.add_argument("--scale", type=float, default=3.0)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--target-faces", type=int, default=50_000)
    a = ap.parse_args()
    run(a.out, a.faces, a.scale, a.views, target_faces=a.target_faces)
    print((Path(a.out) / "bench.md").read_text())
