"""Dark fabrics: PS error and the fabric floor against albedo, at one exposure and with brackets.

The synthetic fabric board (placements at 0, +-30 and +-50 deg) is rendered with its albedo
texture scaled to each mean albedo, through a crop camera at the planned camera's pixel density
on the fabric (default 5.75 px/mm: 50 mm lens, 6 um pixels, 1.45 m), and stage 6 runs on it
(stages 3-5 against the known board) once per albedo and exposure set. The exposure is anchored
to the rig's light budget (CLAUDE.md, Rig -> lights): a 90 % white facing the median light
reaches --anchor of full scale at the PS shutter.

  bench_dark_fabric.py --out runs/dark_fabric --noise sensor
  bench_dark_fabric.py --out runs/dark_fabric_real --noise <stage-C radiometry.json>

--noise: 'test' (the suite's 0.5 % read noise, full well 10,000: pessimistic), 'sensor' (read
1.5e-4, full well 60,000: a guess at a 14-bit full-frame body at ISO 100, as test_R_model_error
uses), or the pilot's radiometry.json, which is the real answer once the camera exists.

Per albedo and exposure set it reports the per-pixel PS error against the truth (unshadowed
pixels with a solve), the share of fabric pixels stage 5 can use (confidence >= 0.3), the mean
number of lights kept, and the fabric floor at s = 5-50 mm. A floor above 3 deg at any s >= 5 mm
makes the verdict INCONCLUSIVE (rule 1); the floor plus 1 deg is the smallest disagreement a
cluster can show. Writes <out>/dark_fabric.json and dark_fabric.md.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

import stage6_floor
from gwps import synth
from gwps.camera import Camera
from gwps.compare import angle_deg

ALBEDOS = [0.5, 0.25, 0.12, 0.06, 0.03, 0.015]
NOISE = {"test": (synth.READ_NOISE, synth.FULL_WELL), "sensor": (1.5e-4, 60000.0)}
ANGLES = [0, 30, -30, 50, -50]
SCALES = (5, 10, 20, 50)


def noise_params(noise):
    if noise in NOISE:
        return NOISE[noise]
    d = json.loads(Path(noise).read_text())
    d = d.get("noise", d)
    return float(d["read"]), float(d["full_well"])


def crop_camera(px_per_mm, margin_px=16):
    """PINHOLE camera at px_per_mm on the board's centre, cropped to the board at every placement."""
    R0, t0 = synth.base_pose()
    z = float(np.linalg.norm(synth.TARGET - synth.CAM_CENTRE))
    f = px_per_mm * 1000.0 * z
    mesh, _ = synth.board_mesh()
    uv = []
    for R, t in synth.poses(ANGLES):
        Xc = mesh.vertices @ R.T + t
        uv.append(f * Xc[:, :2] / Xc[:, 2:])
    uv = np.concatenate(uv)
    lo, hi = uv.min(0) - margin_px, uv.max(0) + margin_px
    w, h = (np.ceil(hi - lo)).astype(int)
    return Camera(1, "PINHOLE", int(w), int(h), f, f, float(-lo[0]), float(-lo[1]))


def anchor_scale(anchor):
    """Factor on albedo (equivalently on exposure) that puts a 90 % white facing the median light
    at `anchor` of full scale, at the board's centre."""
    R0, t0 = synth.base_pose()
    X = (R0 @ synth.TARGET + t0)[None].astype(np.float64)
    lights, _ = synth.studio_lights()
    irr = [float(np.linalg.norm(L.light_vector(X)[0][0])) for L in lights]
    return anchor / (0.9 * float(np.median(irr)))


def texture_mean():
    mesh, _ = synth.board_mesh()
    return float(np.mean(synth.albedo(mesh.triangles_center)))


def measure(d, read, full_well):
    fl = stage6_floor.run(d / "sparse/0", d / "mesh.ply", d / "garment_faces.npy", d / "manifest.csv",
                          d / "lights.json", d / "run", "fabric", read_noise=read, full_well=full_well)
    errs, usable, lights = [], [], []
    for f in sorted((d / "gt").glob("*.npz")):
        g, p = np.load(f), np.load(d / "run/stage4" / f.name)
        sel = g["unshadowed"] & p["ok"]
        errs.append(angle_deg(p["n_ps"][sel], g["n_gt_cam"][sel]))
        usable.append(p["conf"][g["hit"]] >= 0.3)
        lights.append(p["n_lights"][g["hit"]])
    e, u, nl = np.concatenate(errs), np.concatenate(usable), np.concatenate(lights)
    meta = json.loads((d / "run/stage4/meta.json").read_text())
    frac = [np.mean(list(v["bracket_fraction"].values())) for v in meta["views"].values() if "bracket_fraction" in v]
    return {"px_median_deg": float(np.median(e)), "px_p90_deg": float(np.percentile(e, 90)),
            "usable": float(u.mean()), "lights_mean": float(nl.mean()),
            "from_brackets": float(np.mean(frac)) if frac else 0.0,
            "floor_deg": {str(s): float(fl[s]) for s in SCALES}}


def run(out, albedos=ALBEDOS, brackets=(4.0, 16.0), noise="sensor", px_per_mm=5.75, anchor=0.6, seed=0, keep=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    read, fw = noise_params(noise)
    cam = crop_camera(px_per_mm)
    scale0 = anchor_scale(anchor) / texture_mean()
    sets = [("one exposure", (1.0,))] + ([("bracketed", (1.0,) + tuple(map(float, brackets)))] if brackets else [])
    rows = []
    for a in albedos:
        for label, ex in sets:
            t = time.time()
            d = out / f"albedo{a}_{len(ex)}x"
            synth.make_dataset(d, "board", None, noise=True, seed=seed, angles=ANGLES, noise_params=(read, fw),
                               albedo_scale=a * scale0, exposures=ex, cam=cam)
            r = {"albedo": a, "exposures": list(ex), "label": label, **measure(d, read, fw), "seconds": time.time() - t}
            rows.append(r)
            print(f"albedo {a:.3f} {label:12s} px median {r['px_median_deg']:.2f} deg, usable {r['usable']:.3f}, "
                  f"floor 5 mm {r['floor_deg']['5']:.3f} deg ({r['seconds']:.0f} s)", flush=True)
            if not keep:
                for f in list(d.rglob("*.npy")) + list(d.rglob("*.npz")):
                    f.unlink()
    rep = {"noise": noise, "read": read, "full_well": fw, "px_per_mm": px_per_mm, "anchor": anchor,
           "camera": {"width": cam.width, "height": cam.height, "fx": cam.fx}, "brackets": list(brackets or []),
           "rows": rows}
    (out / "dark_fabric.json").write_text(json.dumps(rep, indent=1))
    L = [f"# Dark fabrics: PS error and fabric floor against albedo", "",
         f"Noise: {noise} (read {read:g}, full well {fw:g}); {px_per_mm} px/mm on the fabric; a 90 % white facing "
         f"the median light at {anchor:.0%} of full scale at the PS shutter; brackets x{', x'.join(f'{b:g}' for b in brackets or [])}.", "",
         "| albedo | exposures | PS error median / p90 (deg) | usable | lights | floor 5 / 10 / 20 / 50 mm (deg) | |",
         "|---|---|---|---|---|---|---|"]
    for r in rows:
        f = r["floor_deg"]
        flag = "INCONCLUSIVE" if max(f[str(s)] for s in SCALES) > 3 else ""
        L.append(f"| {r['albedo']:g} | {r['label']} | {r['px_median_deg']:.2f} / {r['px_p90_deg']:.2f} | "
                 f"{100 * r['usable']:.1f} % | {r['lights_mean']:.2f} | "
                 f"{' / '.join(f'{f[str(s)]:.3f}' for s in SCALES)} | {flag} |")
    (out / "dark_fabric.md").write_text("\n".join(L) + "\n")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--albedo", type=float, nargs="+", default=ALBEDOS, help="mean fabric albedos")
    ap.add_argument("--brackets", type=float, nargs="*", default=[4.0, 16.0],
                    help="bracket exposures as multiples of the PS shutter (none: one exposure only)")
    ap.add_argument("--noise", default="sensor", help="test | sensor | radiometry.json")
    ap.add_argument("--px-per-mm", type=float, default=5.75)
    ap.add_argument("--anchor", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", action="store_true", help="keep the rendered images")
    a = ap.parse_args()
    run(a.out, a.albedo, a.brackets, a.noise, a.px_per_mm, a.anchor, a.seed, a.keep)
    print((Path(a.out) / "dark_fabric.md").read_text())
