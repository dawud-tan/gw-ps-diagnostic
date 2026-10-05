"""Stage 3: per-view mesh maps by embree ray casting (COLMAP pixel centres, camera frame).

Writes <out>/<colmap_image_stem>.npz with hit, depth, pos_cam, face_id, normal_cam.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from gwps.camera import load_model
from gwps.io import save_npz, write_meta
from gwps.raycast import Caster, cast_view


def run(sparse, mesh, out, quiet=False):
    cams, images = load_model(sparse)
    m = trimesh.load(mesh, process=False)
    caster = Caster(m)
    out = Path(out)
    per_view = {}
    for im in images.values():
        maps = cast_view(caster, cams[im.camera_id], im.R, im.t)
        hit = maps["hit"]
        back = np.sum(maps["normal_cam"][hit] * maps["pos_cam"][hit], -1) > 0   # n . (C - X) < 0
        frac = float(back.mean()) if hit.any() else 0.0
        per_view[im.name] = {"hits": int(hit.sum()), "back_facing_fraction": frac}
        if frac > 0.01 and not quiet:
            print(f"WARNING {im.name}: {100 * frac:.1f} % back-facing hits -> inconsistent winding?")
        save_npz(out / f"{im.stem}.npz", **maps)
    write_meta(out, {"sparse": sparse, "mesh": mesh}, {}, {"views": per_view})
    return per_view


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sparse", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(json.dumps(run(a.sparse, a.mesh, a.out), indent=1))
