"""M. Object masks: from backlit silhouettes (default) and from SfM frames against a matte black
backdrop, with the static exclusion, against the true masks of a synthetic session; and the
pieces (Otsu, hole filling, feature margins)."""
import numpy as np
import pytest

import stage_masks
from gwps import masks as M
from gwps.capture import develop_for_colmap
from gwps.synth_sfm import make_session


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    d = tmp_path_factory.mktemp("masks")
    lit = make_session(d / "lit", n_steps=3, metric_angles=(), seed=1)
    black = make_session(d / "black", n_steps=3, metric_angles=(), seed=2, backdrop="black")
    for s in (lit, black):
        develop_for_colmap(s)
    return lit, black


def _iou(session, out):
    r = []
    for name in sorted(p.name for p in (session / "truth_masks").iterdir()):
        m, t = M.read_mask(out / name), M.read_mask(session / "truth_masks" / name)
        r.append(((m & t).sum() / (m | t).sum(), (t & ~m).mean(), (m & ~t).mean()))
    return np.array(r)


def test_silhouette_masks(sessions):
    lit, _ = sessions
    stage_masks.run(lit, "silhouette", lit / "static_exclude.png", out=lit / "masks")
    stage_masks.run(lit, "silhouette", None, out=lit / "masks_noexcl")
    r, rn = _iou(lit, lit / "masks"), _iou(lit, lit / "masks_noexcl")
    print(f"\n[M] silhouette: IoU {r[:, 0].min():.4f} (missed {r[:, 1].max():.5f}, extra {r[:, 2].max():.5f}); "
          f"without the exclusion IoU {rn[:, 0].min():.4f} (extra {rn[:, 2].max():.5f}: the static base)")
    assert r[:, 0].min() > 0.985 and r[:, 1].max() < 0.001        # the object is all in; a thin grown edge
    assert rn[:, 2].min() > r[:, 2].max() + 0.003                  # the base leaks in without the exclusion


def test_sfm_masks_black_backdrop(sessions):
    _, black = sessions
    with pytest.raises(ValueError, match="no silhouette"):
        stage_masks.run(black, "silhouette")
    rep = stage_masks.run(black, "sfm", black / "static_exclude.png")
    r = _iou(black, black / "masks")
    print(f"\n[M] SfM frame on black velvet: IoU {r[:, 0].min():.4f} (missed {r[:, 1].max():.5f}, extra {r[:, 2].max():.5f}); "
          f"threshold {min(v['threshold'] for v in rep['views'].values()):.4f}")
    assert r[:, 0].min() > 0.97


def test_mask_pieces():
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0.05, 0.01, 5000), rng.normal(0.8, 0.05, 5000)]
    assert 0.1 < M.otsu(x) < 0.7
    obj = np.zeros((60, 60), bool)
    obj[10:50, 10:50] = True
    obj[25:35, 25:35] = False                                     # a hole: background inside the object
    assert M._fill_holes(obj)[30, 30] and not M._fill_holes(obj)[5, 5]
    img = np.where(obj, 0.02, 0.9).astype(np.float32)            # backlit: holes stay background
    m, _ = M.silhouette_mask(img, dilate_px=0)
    assert not m[30, 30] and m[12, 12] and not m[5, 5]
    fm = M.feature_mask(obj, 3)
    assert fm[12, 12] == 0 and fm[15, 15] == 255                 # features kept 3 px inside the edge
    with pytest.raises(ValueError, match="does not match"):
        M.exclude(obj, np.zeros((10, 10), bool))
