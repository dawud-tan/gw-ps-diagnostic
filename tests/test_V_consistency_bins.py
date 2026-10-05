"""V. Consistency bins: views grouped by turntable angle (camera azimuth about world +Z) for the
mean resultant length, and the accumulator's handling of bins."""
import numpy as np
import pytest

from gwps.camera import Image, R_to_qvec
from gwps.compare import FaceAccumulator, consistency_bins
from gwps.synth import poses


def _images(angles):
    return [Image(k + 1, R_to_qvec(R), t, 1, f"step{k:04d}.png") for k, (R, t) in enumerate(poses(angles))]


def test_bins_follow_the_turntable_angle():
    b10 = consistency_bins(_images([10.0 * k for k in range(36)]), 30)
    assert set(np.bincount(list(b10.values()))) == {3} and len(set(b10.values())) == 12
    assert b10[36] == b10[1] == b10[2]                      # steps 35, 0, 1: bins are centred on step 0
    assert b10[3] == b10[4] == b10[5] != b10[1]
    b30 = consistency_bins(_images([30.0 * k for k in range(12)]), 30)
    assert sorted(b30.values()) == list(range(12))          # 30 deg steps: one view per bin, as validated
    jitter = np.random.default_rng(0).uniform(-3, 3, 36)     # a hand-turned table
    bj = consistency_bins(_images([10.0 * k + j for k, j in enumerate(jitter)]), 30)
    assert set(np.bincount(list(bj.values()))) == {3}
    assert consistency_bins(_images([0, 10, 20]), 0) == {1: 0, 2: 1, 3: 2}   # bin width 0: per view


def _cmp(direction):
    d = np.array([direction, direction], np.float32)
    return {"valid": np.array([True, True]), "cos_view": np.ones(2), "view_azimuth_deg": np.zeros(2),
            "theta": {5: np.ones(2, np.float32)}, "d_world": {5: d}}


def test_accumulator_averages_within_bins():
    fid = np.array([0, 1])
    acc = FaceAccumulator(3, (5,))
    acc.add(fid, _cmp([1, 0, 0]), 0)
    acc.add(fid, _cmp([1, 0, 0]), 0)
    acc.add(fid, _cmp([0, 1, 0]), 1)
    r = acc.result()
    assert r["n_views_5"][0] == 3 and r["n_bins_5"][0] == 2 and r["n_bins_5"][2] == 0
    assert r["mrl_5"][0] == pytest.approx(np.sqrt(2) / 2)   # per view it would be sqrt(5)/3 = 0.745
    per_view = FaceAccumulator(3, (5,))
    for d in ([1, 0, 0], [1, 0, 0], [0, 1, 0]):
        per_view.add(fid, _cmp(d))
    assert per_view.result()["mrl_5"][0] == pytest.approx(np.sqrt(5) / 3)


def test_vertical_fraction():
    from gwps.verdict import vertical_fraction
    v = vertical_fraction(np.array([[0, 0, 1.0], [1.0, 0, 0], [1.0, 0, 1.0], [0, 0, 0]]))
    assert np.allclose(v[:3], [1.0, 0.0, 0.5]) and np.isnan(v[3])
    fid = np.array([0, 1])
    acc = FaceAccumulator(2, (5,))
    acc.add(fid, _cmp([0, 0.6, 0.8]), 0)
    assert np.allclose(acc.result()["dmean_5"][0], [0, 0.6, 0.8], atol=1e-6)   # the mean resultant vector


def test_accumulator_needs_views_grouped_by_bin():
    fid = np.array([0, 1])
    acc = FaceAccumulator(2, (5,))
    acc.add(fid, _cmp([1, 0, 0]), 0)
    acc.add(fid, _cmp([1, 0, 0]), 1)
    with pytest.raises(ValueError, match="grouped by bin"):
        acc.add(fid, _cmp([1, 0, 0]), 0)
