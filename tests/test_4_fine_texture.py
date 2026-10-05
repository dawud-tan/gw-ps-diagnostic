"""4. Fine texture is ignored: PS normals that differ from the mesh only by a 3 px, +-10 deg
ripple give no cluster and theta_s < 0.5 deg for every s >= 5 mm (guards evidence A)."""
import numpy as np
import pytest

from gwps.compare import SCALES_MM


@pytest.mark.parametrize("noise", ["none", "noisy"])
def test_fine_texture_ignored(synth, noise):
    ds, run, res = synth.verdict("ripple", noise)
    _, faces, _ = synth.stage5(ds)
    garment = np.load(ds / "garment_faces.npy")
    print("\n" + (run / "verdict.md").read_text())
    for s in SCALES_MM:
        th = faces[f"theta_{s}"][garment & (faces[f"n_views_{s}"] >= 2)]
        th = th[np.isfinite(th)]
        print(f"[test4 {noise}] s={s} mm median={np.median(th):.3f} p99={np.percentile(th, 99):.3f} "
              f"max={th.max():.3f} deg clusters={len(res['clusters'][s])}")
        if s >= 5:
            assert res["clusters"][s] == []
            assert th.max() < 0.5
    assert res["verdict"] == "BAKE"
