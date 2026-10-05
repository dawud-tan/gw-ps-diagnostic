"""Gaussian Wrapping dry run on a synthetic garment session: everything but the GPU happens here.

  gw_dryrun.py prepare --out runs/gw_dryrun [--steps 36] [--scale 1] [--bump]
      renders a synthetic garment session in the capture formats (gwps.synth_sfm, with the PS
      frames), runs stage_masks -> stage_sfm -> stage 1 (masks as alpha) and the fabric floor,
      and writes <out>/gw_dataset_upload.tar (images/ + sparse/0/) for the GPU pod.
  (on the pod)  bash gw_pod.sh setup; bash gw_pod.sh run gw_dataset out; bash gw_pod.sh pack out
  gw_dryrun.py finish --work runs/gw_dryrun --mesh <raw GW mesh> [--pod-logs <unpacked pod dir>]
      stage 2 -> stage 3 -> garment selection -> stages 4, 5, 7 -> the bake, plus the GW mesh's
      distance to the true surface and the pod's time and memory; writes <work>/finish*/finish.md.

--scale 1 renders 1920 x 1440 (quick; GW at -r 1 sees about its default width); --scale 3.125
renders 6000 x 4500 (24 MP, the planned camera) for GW's real time and memory. --bump puts a 3 mm
bump on the garment: a GW mesh that smooths it away should come out BUILD, one that keeps it BAKE.
The synthetic garment is the torso's side between z = 0.02 and 0.58 m (the ends stand in for the
mannequin and the turntable), so garment selection here is the region cut.
"""
import argparse
import json
import re
import tarfile
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import stage1_gw_prep
import stage2_frame_gate
import stage3_mesh_maps
import stage4_ps
import stage5_compare
import stage6_floor
import stage7_verdict
import stage_bake
import stage_garment
import stage_masks
import stage_sfm
from gwps.capture import develop_for_colmap
from gwps.synth import Rz, make_dataset
from gwps.synth_sfm import make_session, turning_mesh

GARMENT_Z = (0.02, 0.58)


def prepare(out, steps=36, scale=1.0, bump=False, jitter_deg=1.0, seed=3):
    out = Path(out)
    s = make_session(out / "session", n_steps=steps, jitter_deg=jitter_deg, seed=seed, ps=True, bump=bump, scale=scale)
    develop_for_colmap(s)
    stage_masks.run(s, "silhouette", s / "static_exclude.png")
    sfm = stage_sfm.run(s, s / "intrinsics_true.json", s / "board.json", out / "sfm", camera_height_m=0.30,
                        measured_axis_distance_m=1.60)
    gw = stage1_gw_prep.run(out / "sfm/undistorted", out / "sfm/lit/manifest.csv", out / "gw_dataset",
                            masks=out / "sfm/undistorted/masks")
    b = make_dataset(out / "board", "board", None, True, seed=4, resolution_scale=3.0 * scale)
    stage6_floor.run(b / "sparse/0", b / "mesh.ply", b / "garment_faces.npy", b / "manifest.csv", b / "lights.json",
                     out / "board_run", "fabric")
    with tarfile.open(out / "gw_dataset_upload.tar", "w") as tar:          # PNGs are compressed already
        for part in ("images", "sparse"):
            tar.add(out / "gw_dataset" / part, arcname=f"gw_dataset/{part}")
    rep = {"steps": steps, "scale": scale, "bump": bump, "sfm_pass": sfm["pass"], "sfm_gates": sfm["gates"],
           "stage1": gw["gates"], "cameras": gw["cameras"],
           "upload": str(out / "gw_dataset_upload.tar"), "upload_mb": (out / "gw_dataset_upload.tar").stat().st_size / 2 ** 20}
    (out / "prepare.json").write_text(json.dumps(rep, indent=1, default=float))
    return rep


def truth_mesh_in_chain_frame(session, bump):
    """The true turning mesh in stage_sfm's frame (step 0's camera on -Y; jitter moved it)."""
    t = json.loads((Path(session) / "truth.json").read_text())
    C0 = np.array(t["images"]["step0000.png"]["centre"])
    Q = Rz(-(np.degrees(np.arctan2(C0[1], C0[0])) + 90.0))
    m, _ = turning_mesh(bump)
    T = np.eye(4)
    T[:3, :3] = Q
    m.apply_transform(T)
    return m, Q


def surface_distance_mm(mesh, faces, truth, spacing_m=0.001):
    """Signed distance (mm, + outside) from the selected faces' centres to the true surface: to the
    plane of the true triangle under the nearest of dense surface samples. (The distance to the
    nearest sample alone has a floor of about half their spacing: 0.24 mm median for the true
    surface against itself at 0.5 mm.)"""
    n = int(min(6_000_000, truth.area / spacing_m ** 2))
    pts, fidx = trimesh.sample.sample_surface(truth, n, seed=0)
    _, j = cKDTree(pts).query(mesh.triangles_center[faces], workers=-1)
    f = fidx[j]
    return 1000 * np.einsum("ij,ij->i", truth.face_normals[f], mesh.triangles_center[faces] - truth.vertices[truth.faces[f, 0]])


def pod_stats(pod_dir):
    """Peak GPU memory (gpu.csv from nvidia-smi) and wall time / peak host memory (time.txt)."""
    pod_dir, st = Path(pod_dir), {}
    g = pod_dir / "gpu.csv"
    if g.exists():
        used = [float(r.split(",")[1]) for r in g.read_text().splitlines() if r.count(",") >= 2]
        st["peak_vram_gb"] = max(used) / 1024 if used else None
        st["gpu_samples"] = len(used)
    t = pod_dir / "time.txt"
    if t.exists():
        txt = t.read_text()
        m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", txt)
        if m:
            parts = [float(x) for x in m.group(1).split(":")]
            st["wall_h"] = sum(p * 60 ** i for i, p in enumerate(reversed(parts))) / 3600
        m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt)
        if m:
            st["peak_host_ram_gb"] = int(m.group(1)) / 2 ** 20
    return st


def finish(work, mesh, pod_logs=None, out=None, bake=True):
    work = Path(work)
    out = Path(out or work / "finish")
    prep = json.loads((work / "prepare.json").read_text())
    ds, s = work / "gw_dataset", work / "session"
    sp = ds / "sparse/0"
    rep = {"mesh": str(mesh), "prepare": prep}
    s2 = stage2_frame_gate.run(ds, mesh, out / "stage2", manifest=ds / "manifest.csv")
    rep["stage2_pass"] = s2["pass"]
    stage3_mesh_maps.run(sp, mesh, out / "stage3", quiet=True)
    gsel, grep = stage_garment.run(mesh, out / "garment_faces.npy", z_min=GARMENT_Z[0], z_max=GARMENT_Z[1])
    rep["garment_faces"] = grep["garment"]
    stage4_ps.run(sp, ds / "manifest.csv", s / "lights_true.json", out / "stage3", out / "stage4")
    stage5_compare.run(sp, mesh, out / "garment_faces.npy", out / "stage3", out / "stage4", out / "stage5")
    v = stage7_verdict.run(out / "stage5", mesh, out / "garment_faces.npy", work / "board_run/floors.json",
                           out / "verdict.md", sparse=sp, stage3=out / "stage3", stage4=out / "stage4")
    rep["verdict"] = v["verdict"]
    pk = lambda h: None if not h else h["peak_mm"]
    rep["clusters"] = {str(sc): [{"area_mm2": c["area_mm2"], "mrl": c["mrl"], "vertical_fraction": c["vertical_fraction"],
                                  "centroid_m": c["centroid_world_m"], "height_mm": pk(c.get("height_mm")),
                                  "height_fine_mm": pk(c.get("height_fine_mm"))} for c in cl]
                       for sc, cl in v["clusters"].items() if sc >= 5}
    m = trimesh.load(mesh, process=False)
    truth, Q = truth_mesh_in_chain_frame(s, prep["bump"])
    d = surface_distance_mm(m, np.flatnonzero(gsel), truth)
    a = np.abs(d)
    rep["to_true_surface_mm"] = {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)), "max": float(a.max()),
                                 "mean_signed": float(d.mean())}
    if prep["bump"]:
        tj = json.loads((s / "truth.json").read_text())
        c = Q @ np.array(tj["bump_centre_world_m"])
        rep["bump_centre_chain_m"] = c.tolist()
        rep["bump_found"] = any(np.linalg.norm(np.array(cl["centroid_m"]) - c) < 0.03
                                for sc in ("5", "10") for cl in rep["clusters"].get(sc, []))
    if bake:
        b, _ = stage_bake.run(sp, mesh, out / "garment_faces.npy", out / "stage3", out / "stage4", out / "bake/garment")
        rep["bake"] = {k: b[k] for k in ("faces_garment", "faces_decimated", "texel_coverage_by_views")}
    if pod_logs:
        rep["pod"] = pod_stats(pod_logs)
    (out / "finish.json").write_text(json.dumps(rep, indent=1, default=float))
    (out / "finish.md").write_text(finish_markdown(rep))
    return rep


def finish_markdown(rep):
    p, d = rep["prepare"], rep["to_true_surface_mm"]
    L = [f"# GW dry run: {rep['verdict']}", "",
         f"- Session: {p['steps']} steps, scale {p['scale']} ({p['cameras']['1']['width']} x {p['cameras']['1']['height']} after "
         f"stage 1), bump {'on' if p['bump'] else 'off'}; stage_sfm {'passed' if p['sfm_pass'] else 'FAILED'}.",
         f"- Mesh: {rep['mesh']}; garment faces {rep['garment_faces']['faces']:,} ({rep['garment_faces']['area_m2']:.3f} m^2).",
         f"- Garment faces to the true surface: median {d['median']:.2f} mm, p95 {d['p95']:.2f} mm, max {d['max']:.2f} mm "
         f"(mean signed {d['mean_signed']:+.2f} mm, + outside).",
         f"- Stage 2 frame gates: {'pass' if rep['stage2_pass'] else 'FAIL'}."]
    if "bump_found" in rep:
        L.append(f"- Bump (3 mm, sigma 10 mm) flagged at s = 5 or 10 mm: {'yes' if rep['bump_found'] else 'no'}.")
    for sc, cl in rep["clusters"].items():
        for c in cl:
            h = c.get("height_fine_mm")
            L.append(f"  - s = {sc} mm: {c['area_mm2']:.0f} mm^2, MRL {c['mrl'] if c['mrl'] is None else round(c['mrl'], 2)}, "
                     f"vertical {c['vertical_fraction'] if c['vertical_fraction'] is None else round(c['vertical_fraction'], 2)}, "
                     f"implied height {'n/a' if h is None else f'{h:+.2f} mm'} (at 2 mm; + = the surface lies outside the mesh)")
    if "bake" in rep:
        b = rep["bake"]
        L.append(f"- Bake: {b['faces_garment']:,} -> {b['faces_decimated']:,} faces, texel coverage {b['texel_coverage_by_views']:.3f}.")
    if rep.get("pod"):
        q = rep["pod"]
        L.append(f"- Pod: peak VRAM {q.get('peak_vram_gb', float('nan')):.1f} GB, wall {q.get('wall_h', float('nan')):.2f} h, "
                 f"peak host RAM {q.get('peak_host_ram_gb', float('nan')):.1f} GB.")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=36)
    p.add_argument("--scale", type=float, default=1.0)
    p.add_argument("--bump", action="store_true")
    f = sub.add_parser("finish")
    f.add_argument("--work", required=True)
    f.add_argument("--mesh", required=True, help="the raw GW mesh (…_searched.ply)")
    f.add_argument("--pod-logs", default=None, help="the unpacked result of gw_pod.sh pack")
    f.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "prepare":
        r = prepare(a.out, a.steps, a.scale, a.bump)
        print(json.dumps({k: r[k] for k in ("sfm_pass", "upload", "upload_mb")}, indent=1))
    else:
        finish(a.work, a.mesh, a.pod_logs, a.out)
        print((Path(a.out or Path(a.work) / "finish") / "finish.md").read_text())
