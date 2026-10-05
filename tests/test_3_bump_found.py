"""3. A real bump (Gaussian, sigma 10 mm, height 3 mm) missing from the mesh is found:
a cluster at s = 5 mm and s = 10 mm overlapping the bump, mean resultant length >= 0.7.
Run at the original 12 x 30 deg turntable steps and at the garment plan's 36 x 10 deg."""
import json

import numpy as np
import pytest
import trimesh


@pytest.mark.parametrize("steps", [12, 36])
@pytest.mark.parametrize("noise", ["none", "noisy"])
def test_bump_found(synth, noise, steps):
    ds, run, res = synth.verdict("bump", noise, steps)
    print("\n" + (run / "verdict.md").read_text())
    info = json.loads((ds / "scene.json").read_text())
    c = np.array(info["bump_centre_world_m"])
    mesh = trimesh.load(ds / "mesh.ply", process=False)
    near = np.linalg.norm(mesh.triangles_center - c, axis=1) < 2 * info["bump_sigma_m"]
    for s in (5, 10):
        hits = [cl for cl in res["clusters"][s] if near[cl["faces"]].any()]
        assert hits, f"no cluster overlapping the bump at s={s} mm"
        best = max(hits, key=lambda cl: cl["area_mm2"])
        print(f"[test3 {noise} {steps} steps] s={s} mm cluster area={best['area_mm2']:.0f} mm^2 "
              f"theta={best['theta_mean_deg']:.2f} deg MRL={best['mrl']:.3f} "
              f"(views/bins median {best['views_median']:.0f}/{best['bins_median']:.0f})")
        assert best["mrl"] >= 0.7
        far = [cl for cl in res["clusters"][s] if not near[cl["faces"]].any()]
        assert not far, f"clusters away from the bump at s={s} mm: {[round(x['area_mm2']) for x in far]}"
    assert res["verdict"] == "BUILD candidate"
