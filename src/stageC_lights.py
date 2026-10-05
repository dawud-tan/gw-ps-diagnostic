"""Stage C: light calibration from the ball captures in config/calib_manifest.csv.

1. Matte-ball centre at each position from its backlit silhouette (known radius).
2. Mirror-ball highlight per light and position -> reflected ray (the mirror ball sits on
   the same seat, so it shares the matte ball's centre) -> triangulated light position.
3. Per light, E / LED axis / mu fitted to the matte-ball images, with leave-one-position-
   out prediction and an independent light-position cross-check from the shading.
Writes lights.json (camera frame: lights fixed in the studio) and a report.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gwps import lightfit, sphere
from gwps.calib_io import CalibManifest, lit_minus_ambient
from gwps.camera import load_camera_model
from gwps.io import load_linear, write_meta
from gwps.lights import Light, lights_to_json
from gwps.ps import make_noise_model


def ball_centres(man, cam, radius, camera_id):
    out, stats = {}, {}
    for pos in man.positions("matte_ball", camera_id):
        slot = man.slot("matte_ball", pos)
        if "silhouette" not in slot:
            raise ValueError(f"matte_ball {pos}: no silhouette frame")
        sil = load_linear(slot["silhouette"])[0]
        rpx = cam.fx * radius / 1.5
        u0, v0, r0 = sphere.find_dark_blob(sil, 0.3 * rpx, 3.0 * rpx)
        C, st = sphere.fit_sphere_centre(sphere.silhouette_edges(sil, u0, v0, r0), cam, radius)
        out[pos], stats[pos] = C, st
    return out, stats


def run(calib_manifest, camera, ball_radius_m, out, report_dir, camera_id=1, mirror_radius_m=None,
        read_noise=0.005, full_well=10000.0, position_source="mirror", noise=None, model_error=0.005):
    man = CalibManifest(calib_manifest)
    cam = load_camera_model(camera, camera_id)
    mirror_r = mirror_radius_m or ball_radius_m
    noise = make_noise_model(noise, read_noise, full_well, model_error)
    centres, cstats = ball_centres(man, cam, ball_radius_m, camera_id)
    mpos = [p for p in man.positions("mirror_ball", camera_id) if p in centres]
    if len(mpos) < 3:
        raise ValueError(f"need the mirror ball at >= 3 positions with a matte-ball silhouette; have {mpos}")
    light_ids = sorted(set.intersection(*[set(man.slot("mirror_ball", p)["lights"]) for p in mpos]))
    report = {"ball_centres_m": {p: c.tolist() for p, c in centres.items()}, "silhouette": cstats, "lights": {}}
    lights = []
    for lid in light_ids:
        S, r, hl = [], [], {}
        for p in mpos:
            img, _ = lit_minus_ambient(man.slot("mirror_ball", p), lid)
            uv, q = sphere.find_highlight(img, cam, centres[p], mirror_r)
            s_, r_ = sphere.reflected_ray(uv, cam, centres[p], mirror_r)
            S.append(s_)
            r.append(r_)
            hl[p] = {"uv": uv.tolist(), **q}
        P, tri = sphere.triangulate(S, r)
        rep = {"position_m": P.tolist(), "triangulation": tri, "highlights": hl}
        samples = []
        for p in man.positions("matte_ball", camera_id):
            slot = man.slot("matte_ball", p)
            raw, _ = load_linear(slot["lights"][lid])
            amb = load_linear(slot["ambient"])[0] if "ambient" in slot else None
            samples.append(lightfit.ball_samples(raw, amb, cam, centres[p], ball_radius_m))
        f = lightfit.fit_light(samples, P, noise, fit_position=True)
        P_shading = P + f.position_shift_mm / 1000
        if position_source == "shading":
            P = P_shading
            f = lightfit.fit_light(samples, P, noise, fit_position=False)
            f.position_shift_mm = np.zeros(3)
        rep.update({"E": f.E, "axis": f.axis.tolist(), "mu": f.mu, "sigma_E_rel": f.sigma_E_rel,
                    "sigma_mu": f.sigma_mu, "fit_rms_rel": f.rms_rel, "n_px": f.n_px,
                    "lopo_rel": f.lopo_rel, "position_used_m": P.tolist(), "position_source": position_source,
                    "position_shading_m": P_shading.tolist(),
                    "mirror_vs_shading_mm": float(np.linalg.norm(np.array(rep["position_m"]) - P_shading) * 1000)})
        report["lights"][lid] = rep
        lights.append(Light(lid, "ps", "point", P, f.E, f.axis, f.mu))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(out.read_text()) if out.exists() else {"cameras": []}
    cfg["cameras"] = [c for c in cfg["cameras"] if c["camera_id"] != camera_id]
    cfg["cameras"] += lights_to_json({camera_id: ("camera", lights)})["cameras"]
    out.write_text(json.dumps(cfg, indent=1))
    rd = Path(report_dir)
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "stageC_lights.json").write_text(json.dumps(report, indent=1, default=float))
    (rd / "stageC_lights.md").write_text(markdown(report))
    write_meta(rd, {"calib_manifest": calib_manifest, "camera": camera}, {
        "ball_radius_m": ball_radius_m, "mirror_radius_m": mirror_r, "camera_id": camera_id,
        "position_source": position_source,
        "noise_model": noise.to_dict()})
    return lights, report


def markdown(rep):
    L = ["# Stage C: light calibration", "", "## Ball centres (from backlit silhouettes)", "",
         "| position | centre (m, camera frame) | edge points used | edge fit RMS (px) | image radius (px) |",
         "|---|---|---|---|---|"]
    for p, c in rep["ball_centres_m"].items():
        s = rep["silhouette"][p]
        L.append(f"| {p} | {np.round(c, 4).tolist()} | {s['inliers']}/{s['n_edges']} | {s['rms_px']:.3f} | {s['image_radius_px']:.1f} |")
    L += ["", "## Lights", "",
          "| light | mirror position (m) | ray distance max (mm) | 1-sigma worst axis (mm) | E (x primer albedo) | mu | "
          "leave-one-out max error (%) | mirror vs shading (mm) |", "|---|---|---|---|---|---|---|---|"]
    for lid, r in rep["lights"].items():
        tri = r["triangulation"]
        lo = max(abs(x) for x in r["lopo_rel"]) * 100 if r["lopo_rel"] else float("nan")
        L.append(f"| {lid} | {np.round(r['position_m'], 4).tolist()} | {max(tri['ray_distance_mm']):.2f} | "
                 f"{tri['sigma_worst_axis_mm']:.2f} | {r['E']:.4f} | {r['mu']:.2f} +- {r['sigma_mu']:.2f} | {lo:.2f} | "
                 f"{r['mirror_vs_shading_mm']:.1f} |")
    src = next(iter(rep["lights"].values()))["position_source"] if rep["lights"] else "mirror"
    L += ["", f"Positions written to lights.json come from the **{src}** estimate.",
          "Targets: light position <= 5-10 mm (evidence B: 10 mm ~ 0.5 deg, 20 mm ~ 1.3 deg); E within 2 %.",
          "'mirror vs shading' compares two independent position estimates: the mirror-ball triangulation (geometry",
          "only; assumes the mirror ball sits exactly where the matte ball's silhouette puts it) and a refit of the",
          "position from the matte ball's shading (assumes Lambertian primer). Above ~5 mm, check the ball seat",
          "repeatability, the radii, and the primer."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calib-manifest", required=True)
    ap.add_argument("--camera", required=True, help="cameras.txt (PINHOLE, undistorted images) or stage-C intrinsics JSON (OPENCV, raw images)")
    ap.add_argument("--camera-id", type=int, default=1)
    ap.add_argument("--ball-radius-m", type=float, required=True, help="matte ball radius (callipers)")
    ap.add_argument("--mirror-radius-m", type=float, default=None, help="mirror ball radius if different")
    ap.add_argument("--out", required=True, help="lights.json to write (other cameras in it are kept)")
    ap.add_argument("--report-dir", required=True)
    ap.add_argument("--noise", default=None, help="stage-C radiometry.json (measured noise model)")
    ap.add_argument("--read-noise", type=float, default=0.005, help="used only without --noise")
    ap.add_argument("--full-well", type=float, default=10000.0, help="used only without --noise")
    ap.add_argument("--model-error", type=float, default=0.005)
    ap.add_argument("--position-source", choices=["mirror", "shading"], default="mirror",
                    help="which light-position estimate goes into lights.json (both are reported)")
    a = ap.parse_args()
    run(a.calib_manifest, a.camera, a.ball_radius_m, a.out, a.report_dir, a.camera_id, a.mirror_radius_m,
        a.read_noise, a.full_well, a.position_source, a.noise, a.model_error)
    print((Path(a.report_dir) / "stageC_lights.md").read_text())
