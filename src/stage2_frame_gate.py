"""Stage 2: frame gates on the GW dataset, before trusting any per-pixel comparison. Stop if
any fails. (a) reprojection with our projection code: median distance to COLMAP's POINTS2D
within 0.2 px of COLMAP's mean reprojection error, and mean signed residual < 0.1 px per axis
(catches half-pixel shifts); (b) mesh depth versus sparse-point depth: median |dz| a few mm;
(c) lit images aligned with the step's SfM image: the lit set as a whole within 0.2 px (SfM image
against the sum of the lit frames; accurate to ~0.01 px), and each frame within 0.3 px of the
sum of the others (single-light shading biases that estimate by up to ~0.15 px, so single-frame
bumps are caught from ~0.4 px), and each exposure bracket of a dark garment against the base
frames scaled and clipped like it (gates.bracket_alignment: the set within 0.2 px, each frame
within 0.3 px of its own light's base frame); (d) silhouette overlays (mesh silhouette red; the
object mask, when stage 1 wrote it as alpha, green, with the fraction of mesh pixels outside it
reported).
Writes stage2.json, stage2.md and overlays/; exit code 1 if a gate fails.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

from gwps import gates
from gwps.camera import load_colmap_dir
from gwps.io import Manifest, load_linear
from gwps.raycast import Caster, cast_view

TOL = {"reproj_median_px": 0.2, "reproj_signed_px": 0.1, "depth_median_m": 0.005, "align_px": 0.2, "align_light_px": 0.3}


def run(dataset, mesh, out, manifest=None, tol=None):
    tol = {**TOL, **(tol or {})}
    dataset, out = Path(dataset), Path(out)
    sp = dataset / "sparse"
    sp = sp / "0" if (sp / "0").exists() else sp
    cams, images, points = load_colmap_dir(sp)
    caster = Caster(trimesh.load(mesh, process=False))
    man = Manifest(manifest) if manifest else None
    views, ok_all = {}, True
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    for im in images.values():
        cam = cams[im.camera_id]
        v = {}
        r = gates.reprojection(cam, im, points)
        if r:
            v["reprojection"] = {k: r[k] for k in ("n", "median_px", "mean_signed_px", "colmap_mean_error_px")}
            v["reprojection"]["pass"] = gates.reprojection_pass(r, tol["reproj_median_px"], tol["reproj_signed_px"])
            d = gates.depth_check(caster, cam, im, r["Xc"], r["uv"])
            if d:
                v["depth"] = {**d, "pass": d["median_abs_m"] <= tol["depth_median_m"]}
        sfm_path = dataset / "images" / im.name
        sfm = cv2.imread(str(sfm_path), cv2.IMREAD_UNCHANGED)
        alpha = None
        if sfm is not None and sfm.ndim == 3 and sfm.shape[2] == 4:      # stage 1 wrote the object mask as alpha
            alpha, sfm = sfm[..., 3] > 127, sfm[..., :3]
        if sfm is not None and man and im.name in man.views and man.views[im.name]["lights"]:
            view = man.views[im.name]
            sfm_lin = gates.linear_from_image(sfm)
            amb = load_linear(view["ambient"])[0] if view["ambient"] is not None else 0.0
            lit = {lid: load_linear(pth)[0] - amb for lid, pth in sorted(view["lights"].items())}
            al = gates.lit_alignment(sfm_lin, lit, tol["align_light_px"])
            set_ok = al["set"] is not None and np.hypot(*al["set"]) < tol["align_px"]
            bad = [str(k) for k in al["flagged"]]
            v["alignment"] = {"set_shift_px": al["set"], "per_light_px": {str(k): s for k, s in al["per_light"].items()},
                              "pass": bool(set_ok and not bad), "failing_lights": bad}
            # exposure brackets (dark garments): stage 4 mixes them per pixel with the base frames,
            # so a camera bump between the PS shutter and a bracket would mix two positions. Each
            # bracket frame is checked against the base frame of the same light, scaled and clipped
            # like it (gates.bracket_alignment).
            brs = []
            if view.get("brackets"):
                base_raw = {lid: load_linear(pth)[0] for lid, pth in sorted(view["lights"].items())}
            for b in view.get("brackets", []):
                raw_b, sat_b = {}, {}
                for lid, pth in sorted(b["lights"].items()):
                    raw_b[lid], sat_b[lid] = load_linear(pth)
                ab = gates.bracket_alignment(base_raw, raw_b, sat_b, b["exposure_s"] / view["exposure_s"],
                                             tol["align_px"], tol["align_light_px"])
                brs.append({"exposure_s": b["exposure_s"], "set_shift_px": ab["set"],
                            "per_light_px": {str(k): s for k, s in ab["per_light"].items()},
                            "failing_lights": [str(k) for k in ab["flagged"]],
                            "pass": bool(ab["set_pass"] and not ab["flagged"])})
                del raw_b, sat_b
            if brs:
                v["alignment"]["brackets"] = brs
                v["alignment"]["pass"] = bool(v["alignment"]["pass"] and all(b["pass"] for b in brs))
        if sfm is not None:
            maps = cast_view(caster, cam, im.R, im.t)
            img8 = sfm if sfm.dtype == np.uint8 else cv2.convertScaleAbs(sfm, alpha=255.0 / max(float(sfm.max()), 1))
            cv2.imwrite(str(out / "overlays" / f"{im.stem}.png"), gates.silhouette_overlay(img8, maps["hit"], mask=alpha))
            if alpha is not None:                     # mesh outside every mask would be carved away by GW
                v["mask"] = {"mesh_hits_outside_mask": float((maps["hit"] & ~alpha).sum() / max(maps["hit"].sum(), 1))}
        v["pass"] = all(x.get("pass", True) for x in v.values() if isinstance(x, dict))
        ok_all &= v["pass"]
        views[im.name] = v
    rep = {"pass": bool(ok_all), "tolerances": tol, "views": views}
    (out / "stage2.json").write_text(json.dumps(rep, indent=1, default=float))
    lines = ["# Stage 2 frame gates: " + ("PASS" if ok_all else "**FAIL**"), "",
             "| view | reproj median (px) | COLMAP mean err (px) | mean signed (px) | depth median abs (mm) | lit-set shift (px) | max per-light shift (px; brackets: against the base frame) | |",
             "|---|---|---|---|---|---|---|---|"]
    for name, v in views.items():
        rp, dp, al = v.get("reprojection", {}), v.get("depth", {}), v.get("alignment", {})
        per = list(al.get("per_light_px", {}).values()) + [x for b in al.get("brackets", []) for x in b["per_light_px"].values()]
        ms = max((np.hypot(*s) for s in per if s), default=float("nan"))
        ss = np.hypot(*al["set_shift_px"]) if al.get("set_shift_px") else float("nan")
        lines.append(f"| {name} | {rp.get('median_px', float('nan')):.3f} | {rp.get('colmap_mean_error_px', float('nan')):.3f} | "
                     f"{np.round(rp.get('mean_signed_px', [np.nan, np.nan]), 3).tolist()} | {1000 * dp.get('median_abs_m', float('nan')):.2f} | "
                     f"{ss:.3f} | {ms:.3f} | {'ok' if v['pass'] else '**FAIL**'} |")
    (out / "stage2.md").write_text("\n".join(lines) + "\n")
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="GW dataset from stage 1 (images/, sparse/0/)")
    ap.add_argument("--mesh", required=True, help="raw GW mesh")
    ap.add_argument("--manifest", default=None, help="lit manifest (stage 1 output)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = run(a.dataset, a.mesh, a.out, a.manifest)
    print((Path(a.out) / "stage2.md").read_text())
    sys.exit(0 if r["pass"] else 1)
