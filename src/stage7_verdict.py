"""Stage 7: verdict from per-face theta_s (stage 5) and the control floors (stage 6).

With --sparse, --stage3 and --stage4 (the inputs stage 5 compared), each cluster at s >= 5 mm
also gets its implied height error in mm (gwps.height: the low-passed normal difference
integrated over the views that see it within 45 deg of face-on; > 0 = the surface PS sees lies
outside the mesh): height_mm at the cluster's own scale, and height_fine_mm with the same region
integrated at 2 mm, which comes closer to the full depth (a missing 3 mm bump of sigma 10 mm:
2.15 mm at s = 5 mm, 2.61 mm at 2 mm).
"""
import argparse
import json
from dataclasses import fields
from pathlib import Path

import numpy as np
import trimesh

from gwps.camera import load_model
from gwps.compare import SCALES_MM, CompareParams
from gwps.height import cluster_heights
from gwps.verdict import VERDICT_MIN_SCALE_MM, decide, verdict_markdown


def stage5_params(stage5):
    """The CompareParams stage 5 ran with (its meta.json), so heights use the same valid pixels."""
    meta = Path(stage5) / "meta.json"
    d = json.loads(meta.read_text()).get("params", {}) if meta.exists() else {}
    known = {f.name for f in fields(CompareParams)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in known}
    return CompareParams(**kw)


FINE_MM = 2


def add_heights(res, sparse, stage3, stage4, garment, params):
    """Annotate each cluster at s >= 5 mm with height_mm (at its scale) and height_fine_mm (at
    FINE_MM); None where no view sees it face-on enough, with a ring around it."""
    todo = [((s, k, key), c["faces"], s if key == "height_mm" else FINE_MM)
            for s in res["clusters"] if s >= VERDICT_MIN_SCALE_MM
            for k, c in enumerate(res["clusters"][s]) for key in ("height_mm", "height_fine_mm")]
    if not todo:
        return res
    cams, images = load_model(sparse)

    def views():
        for im in images.values():
            m = dict(np.load(Path(stage3) / f"{im.stem}.npz"))
            p = np.load(Path(stage4) / f"{im.stem}.npz")
            yield m, {"n": p["n_ps"], "conf": p["conf"], "ok": p["ok"]}, cams[im.camera_id].fx
            del m, p

    hts = cluster_heights(views(), garment, todo, params)
    for (s, k, key), h in hts.items():
        res["clusters"][s][k][key] = h
    return res


def run(stage5, mesh, garment_faces, floors, out_md=None, scales=SCALES_MM, sparse=None, stage3=None, stage4=None):
    m = trimesh.load(mesh, process=False)
    faces = dict(np.load(Path(stage5) / "faces.npz"))
    garment = np.load(garment_faces).astype(bool)
    fl = json.loads(Path(floors).read_text()) if not isinstance(floors, dict) else floors
    fabric = {int(k): float(v) for k, v in fl["fabric"].items()}
    sphere = {int(k): float(v) for k, v in fl["sphere"].items()} if "sphere" in fl else None
    res = decide(m, faces, garment, fabric, scales, sphere, fl.get("fabric_placements"))
    if sparse and stage3 and stage4:
        add_heights(res, sparse, stage3, stage4, garment, stage5_params(stage5))
    if out_md:
        Path(out_md).write_text(verdict_markdown(res, scales))
        js = {k: v for k, v in res.items() if k != "clusters"}
        js["clusters"] = {str(s): [{k: v for k, v in c.items() if k != "faces"} for c in cl]
                          for s, cl in res["clusters"].items()}
        Path(out_md).with_suffix(".json").write_text(json.dumps(js, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("stage5", "mesh", "garment-faces", "floors", "out"):
        ap.add_argument(f"--{k}", required=True)
    for k in ("sparse", "stage3", "stage4"):
        ap.add_argument(f"--{k}", default=None, help="stage 5's inputs: with all three, clusters get height_mm")
    a = ap.parse_args()
    r = run(a.stage5, a.mesh, a.garment_faces, a.floors, a.out, sparse=a.sparse, stage3=a.stage3, stage4=a.stage4)
    print(Path(a.out).read_text())
