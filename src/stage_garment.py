"""Garment face selection for the raw GW mesh (criteria in gwps/garment.py).

  stage_garment.py --mesh <gw_out>/mesh_..._searched.ply --out runs/shirt01/garment_faces.npy \
                   [--reference runs/mannequin/mesh_bare.ply] [--z-min 0.01] [--z-max 1.2] [--r-max 0.6] \
                   [--garment-masks DIR --stage3 runs/shirt01/stage3]

The mesh must be in the turntable frame: the GW dataset from stage 1 is (stage_sfm.py), and so is
a bare-mannequin reference processed the same way. --garment-masks: 8-bit PNGs named like the
views (step0003.png), 255 = garment, in the geometry of stage 3's views (the GW dataset's images);
any subset of views will do. Writes <out> (bool per face), <out stem>.ply (garment red, the rest
blue, for a look in any mesh viewer) and <out stem>.json (what each criterion kept or dropped,
and the registration).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh

from gwps import garment as G
from gwps.ply import write_face_scalar_ply


def load_masks(mask_dir, stage3_dir):
    masks, maps = {}, {}
    for p in sorted(Path(mask_dir).glob("*.png")):
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        s3 = Path(stage3_dir) / f"{p.stem}.npz"
        if not s3.exists():
            raise ValueError(f"{p.name}: no stage 3 maps at {s3}")
        masks[p.name] = m > 127
        maps[p.name] = np.load(s3)["face_id"]
    return maps, masks


def run(mesh, out, reference=None, z_min=0.01, z_max=None, r_max=0.6, ref_dist_mm=2.0, register=True,
        garment_masks=None, stage3=None, min_component_frac=0.01):
    m = trimesh.load(mesh, process=False)
    ref = trimesh.load(reference, process=False) if reference else None
    maps, masks = load_masks(garment_masks, stage3) if garment_masks else (None, None)
    sel, rep = G.select(m, z_min, z_max, r_max, ref, ref_dist_mm / 1000.0, register, maps, masks, min_component_frac)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, sel)
    write_face_scalar_ply(out.with_suffix(".ply"), m.vertices, m.faces, sel.astype(np.float32), name="garment", vmax=1.0)
    rep["inputs"] = {"mesh": str(mesh), "reference": str(reference) if reference else None,
                     "garment_masks": str(garment_masks) if garment_masks else None}
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1, default=float))
    return sel, rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True, help="garment_faces.npy")
    ap.add_argument("--reference", default=None, help="bare-mannequin mesh, turntable frame")
    ap.add_argument("--z-min", type=float, default=0.01, help="m above the turntable top")
    ap.add_argument("--z-max", type=float, default=None)
    ap.add_argument("--r-max", type=float, default=0.6, help="m from the turntable axis")
    ap.add_argument("--ref-dist-mm", type=float, default=2.0)
    ap.add_argument("--no-register", action="store_true", help="the reference is already aligned")
    ap.add_argument("--garment-masks", default=None)
    ap.add_argument("--stage3", default=None, help="stage 3 output (face-id maps) for --garment-masks")
    ap.add_argument("--min-component-frac", type=float, default=0.01)
    a = ap.parse_args()
    if a.garment_masks and not a.stage3:
        ap.error("--garment-masks needs --stage3")
    _, r = run(a.mesh, a.out, a.reference, a.z_min, a.z_max, a.r_max, a.ref_dist_mm, not a.no_register,
               a.garment_masks, a.stage3, a.min_component_frac)
    print(json.dumps({k: r[k] for k in ("region", "garment") if k in r}, indent=1))
