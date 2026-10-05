"""Stage 6: control captures -> floors. Runs stages 3-4 and the stage 5 comparison with
the known control geometry (board or sphere mesh) in place of the GW mesh.

floor_s = 95th percentile of theta_s over valid control pixels. No per-view rotation is
removed for controls: on a board a rotation is indistinguishable from a uniform sheen
bias, and the floor must include that bias.

Per placement (control view), floors.json also gets <kind>_placements: the elevation span it
covers in the camera (Y / Z of its valid pixels, 1st-99th percentile; see compare.compare_view)
and its own floor per scale. Stage 7 checks the garment's elevations against these spans: a
light-fixed bias that tilts normals with the image row passes the consistency test (the known
blind spot), and only a board placement at that height can show it.
"""
import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

import stage3_mesh_maps
import stage4_ps
from gwps.compare import CompareParams
from gwps.verdict import floor_from_thetas
from stage5_compare import iter_view_comparisons


def run(sparse, mesh, garment_faces, manifest, lights, run_dir, kind="fabric", params=None,
        floors_path=None, read_noise=0.005, full_well=10000.0, noise=None, model_error=0.005):
    params = replace(params or CompareParams(), remove_rotation=False)
    run_dir = Path(run_dir)
    s3, s4 = run_dir / "stage3", run_dir / "stage4"
    stage3_mesh_maps.run(sparse, mesh, s3, quiet=True)
    stage4_ps.run(sparse, manifest, lights, s3, s4, read_noise=read_noise, full_well=full_well,
                  noise=noise, model_error=model_error)
    garment = np.load(garment_faces).astype(bool)
    thetas = {s: [] for s in params.scales_mm}
    placements = []
    rng = np.random.default_rng(0)
    for im, _, cmp in iter_view_comparisons(sparse, s3, s4, garment, params):
        pl = {"name": Path(im.name).stem, "elev": None, "floor_deg": {}}
        if len(cmp["elev"]):
            pl["elev"] = [float(x) for x in np.percentile(cmp["elev"], [1, 99])]
        for s in params.scales_mm:
            t = cmp["theta"].get(s)
            if t is not None:
                t = t[np.isfinite(t)]
                if len(t) > 2_000_000:                    # a 95th percentile needs no more (24 MP views)
                    t = rng.choice(t, 2_000_000, replace=False)
                thetas[s].append(t)
                if len(t):
                    pl["floor_deg"][str(s)] = float(np.percentile(t, 95))
        placements.append(pl)
        del cmp                                           # before the next view is compared
    floor = floor_from_thetas({s: np.concatenate(v) for s, v in thetas.items()})
    floors_path = Path(floors_path or run_dir / "floors.json")
    allf = json.loads(floors_path.read_text()) if floors_path.exists() else {}
    allf[kind] = {str(s): v for s, v in floor.items()}
    allf[f"{kind}_placements"] = placements
    floors_path.parent.mkdir(parents=True, exist_ok=True)
    floors_path.write_text(json.dumps(allf, indent=1))
    return floor


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for k in ("sparse", "mesh", "garment-faces", "manifest", "lights", "run-dir"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--kind", choices=["fabric", "sphere"], default="fabric")
    ap.add_argument("--floors", default=None)
    a = ap.parse_args()
    print(json.dumps(run(a.sparse, a.mesh, a.garment_faces, a.manifest, a.lights, a.run_dir, a.kind,
                         floors_path=a.floors), indent=1))
