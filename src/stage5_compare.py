"""Stage 5: compare mesh and PS normals per view (camera frame, garment faces only),
low-pass both as vectors at each scale, accumulate per face over views. For the consistency
statistic, views are grouped into turntable-angle bins (CompareParams.consistency_bin_deg,
30 deg): the model must be in the turntable frame (+Z along the turntable axis).

Writes <out>/faces.npz, heatmap_theta_<s>mm.ply, summary.json, summary.md.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from gwps.camera import load_model
from gwps.compare import CompareParams, FaceAccumulator, compare_view, consistency_bins
from gwps.io import write_meta
from gwps.ply import write_face_scalar_ply


def iter_view_comparisons(sparse, stage3, stage4, garment, params, order=None):
    """Yield (image, mesh maps, comparison) per view, in `order` (image ids) if given. A view's
    arrays are released before the next is loaded (callers should `del` theirs too): at 24 MP one
    view holds ~2 GB of inputs and outputs, and keeping the last one alive cost ~2 GB of peak."""
    cams, images = load_model(sparse)
    for iid in (order if order is not None else list(images)):
        im = images[iid]
        m = dict(np.load(Path(stage3) / f"{im.stem}.npz"))
        p = np.load(Path(stage4) / f"{im.stem}.npz")
        ps_out = {"n": p["n_ps"], "conf": p["conf"], "ok": p["ok"]}
        cmp = compare_view(m, ps_out, garment, im.R, cams[im.camera_id].fx, params)
        del ps_out, p
        yield im, m, cmp
        del m, cmp


def run(sparse, mesh, garment_faces, stage3, stage4, out, params=None):
    params = params or CompareParams()
    m = trimesh.load(mesh, process=False)
    garment = np.load(garment_faces).astype(bool)
    assert len(garment) == len(m.faces), "garment_faces.npy does not match the mesh"
    acc = FaceAccumulator(len(m.faces), params.scales_mm)
    _, images = load_model(sparse)
    bins = consistency_bins(images.values(), params.consistency_bin_deg)
    order = sorted(images, key=lambda i: (bins[i], images[i].name))
    per_view, pooled = {}, {s: [] for s in params.scales_mm}
    rng = np.random.default_rng(0)
    for im, maps, cmp in iter_view_comparisons(sparse, stage3, stage4, garment, params, order):
        acc.add(maps["face_id"], cmp, bins[im.image_id])
        v = {"valid_px": int(cmp["valid"].sum()), "rotation_deg": cmp["rotation_deg"],
             "consistency_bin": bins[im.image_id]}
        for s in params.scales_mm:
            th = cmp["theta"].get(s)
            if th is not None and np.isfinite(th).any():
                t = th[np.isfinite(th)]
                v[f"theta_{s}mm_median_deg"] = float(np.median(t))
                v[f"theta_{s}mm_p95_deg"] = float(np.percentile(t, 95))
                # pooled median / p95 need no more: all pixels of 36 views at 24 MP would be ~7 GB
                pooled[s].append(t if len(t) <= 200_000 else rng.choice(t, 200_000, replace=False))
        per_view[im.name] = v
        del maps, cmp
    res = acc.result()
    res["consistency_bin_deg"] = np.float64(params.consistency_bin_deg or 0.0)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "faces.npz", **res)
    elig = garment & (res["n_views_any"] >= 2)
    summary = {"views": per_view, "pooled": {}, "faces": {}}
    for s in params.scales_mm:
        t = np.concatenate(pooled[s]) if pooled[s] else np.array([np.nan])
        summary["pooled"][f"{s}mm"] = {"median_deg": float(np.nanmedian(t)), "p95_deg": float(np.nanpercentile(t, 95))}
        f = res[f"theta_{s}"][elig]
        f = f[np.isfinite(f)]
        summary["faces"][f"{s}mm"] = {"eligible_faces": int(f.size),
                                      "median_deg": float(np.median(f)) if f.size else None,
                                      "max_deg": float(f.max()) if f.size else None}
        write_face_scalar_ply(out / f"heatmap_theta_{s}mm.ply", m.vertices, m.faces,
                              np.where(garment, res[f"theta_{s}"], np.nan))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["# Stage 5 summary", "", "| s (mm) | pooled median (deg) | pooled p95 (deg) | faces >=2 views | face max (deg) |", "|---|---|---|---|---|"]
    for s in params.scales_mm:
        pl, fc = summary["pooled"][f"{s}mm"], summary["faces"][f"{s}mm"]
        lines.append(f"| {s} | {pl['median_deg']:.3f} | {pl['p95_deg']:.3f} | {fc['eligible_faces']} | "
                     f"{fc['max_deg'] if fc['max_deg'] is None else round(fc['max_deg'], 3)} |")
    rots = [v["rotation_deg"] for v in per_view.values()]
    lines += ["", f"Per-view Kabsch rotation: median {np.median(rots):.3f} deg, max {np.max(rots):.3f} deg."]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    write_meta(out, {"sparse": sparse, "mesh": mesh, "garment_faces": garment_faces, "stage3": stage3,
                     "stage4": stage4}, params.__dict__)
    return res, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("sparse", "mesh", "garment-faces", "stage3", "stage4", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    run(a.sparse, a.mesh, a.garment_faces, a.stage3, a.stage4, a.out)
    print((Path(a.out) / "summary.md").read_text())
