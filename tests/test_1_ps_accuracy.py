"""1. PS accuracy: median normal error < 0.1 deg without noise and < 1 deg with noise,
in unshadowed, non-grazing pixels (view angle < 75 deg), against ground truth."""
import numpy as np
import pytest

from gwps.compare import angle_deg

LIMIT = {"none": 0.1, "noisy": 1.0}


def ps_errors(synth, noise):
    ds = synth.dataset("torso", None, noise)
    run = synth.stages34(ds)
    errs = []
    for f in sorted((ds / "gt").glob("*.npz")):
        g, p, m = np.load(f), np.load(run / "stage4" / f.name), np.load(run / "stage3" / f.name)
        X = m["pos_cam"]
        cv = np.sum(m["normal_cam"] * -X / np.maximum(np.linalg.norm(X, axis=-1, keepdims=True), 1e-9), -1)
        sel = g["unshadowed"] & (cv > np.cos(np.radians(75))) & p["ok"]
        errs.append(angle_deg(p["n_ps"][sel], g["n_gt_cam"][sel]))
    return np.concatenate(errs)


@pytest.mark.parametrize("noise", ["none", "noisy"])
def test_ps_accuracy(synth, noise):
    e = ps_errors(synth, noise)
    print(f"\n[test1 {noise}] px={e.size} median={np.median(e):.4f} deg p90={np.percentile(e, 90):.4f} "
          f"p99={np.percentile(e, 99):.4f} deg")
    assert np.median(e) < LIMIT[noise]
