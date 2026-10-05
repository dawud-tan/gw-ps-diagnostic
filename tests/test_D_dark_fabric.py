"""D. Dark fabrics: exposure brackets merged per pixel (gwps.io.merge_brackets), the noise model
of a merged value, the manifest's exposure_s column, stage 2's check of the bracket frames, and a
dark board that one exposure cannot judge and a bracket can."""
import csv
import json

import numpy as np
import pytest

import stage1_gw_prep
import stage2_frame_gate
import stage6_floor
from gwps import ps
from gwps.compare import angle_deg
from gwps.io import BRACKET_MARGIN, Manifest, merge_brackets
from gwps.synth import make_dataset


def test_noise_model_of_a_merged_value():
    m = ps.NoiseModel(read=0.005, full_well=10000.0)
    I = np.array([0.01, 0.2])
    assert np.array_equal(m.var(I), m.var(I, 1.0))                    # k = 1: a single exposure, unchanged
    assert m.var(I, 4.0) == pytest.approx(2 * 0.005 ** 2 / 16 + I / 40000.0)
    m.rel = 0.005
    assert m.var_total(I, 4.0) - m.var(I, 4.0) == pytest.approx((0.005 * I) ** 2)   # model error: relative


def test_merge_takes_the_longest_frame_that_does_not_clip():
    true = np.tile(np.geomspace(0.004, 0.6, 400, dtype=np.float32), (40, 1))       # base frame, noise-free
    amb = np.float32(0.002)
    frames = {k: np.minimum(k * (true + amb), 1.0) for k in (1.0, 4.0, 16.0)}
    sat = {k: f >= 0.98 for k, f in frames.items()}
    I, s, kmap = merge_brackets(frames[1.0], amb, sat[1.0],
                                [(k, frames[k], sat[k], k * amb) for k in (16.0, 4.0)])   # any order
    assert np.allclose(I, true, rtol=1e-5, atol=1e-7)                  # base units, ambient removed
    assert not s.any() and set(np.unique(kmap)) == {1.0, 4.0, 16.0}
    for k in (4.0, 16.0):
        assert (frames[k][kmap == k] < BRACKET_MARGIN).all()           # never a clipped frame
    dark = (true + amb) * 16 < 0.5 * BRACKET_MARGIN
    assert (kmap[dark] == 16).all()                                     # the longest wherever it fits
    with pytest.raises(ValueError, match="longer"):
        merge_brackets(frames[1.0], amb, sat[1.0], [(0.5, frames[1.0], sat[1.0], amb)])


@pytest.mark.parametrize("mu", [0.215, 0.222, 0.228])
def test_merge_has_no_selection_bias_near_the_switch(mu):
    """A uniform patch whose 4x frame lands at the margin. Deciding from the pixel's own base
    value keeps the base exactly where noise pushed it up; the merge decides from the smoothed
    neighbourhood, so its mean stays on the truth."""
    rng = np.random.default_rng(1)
    shape, sb = (300, 300), 0.01
    base = (mu + rng.normal(0, sb, shape)).astype(np.float32)
    long_ = np.minimum(4 * mu + rng.normal(0, 2 * sb, shape), 1.0).astype(np.float32)
    I, _, kmap = merge_brackets(base, 0.0, base >= 0.98, [(4.0, long_, long_ >= 0.98, 0.0)])
    own = base * 4 < BRACKET_MARGIN                                     # the rejected rule, for contrast
    naive = np.where(own, long_ / 4, base)
    bias, naive_bias = float(I.mean() - mu), float(naive.mean() - mu)
    print(f"\n[D] mu {mu}: merged bias {bias:+.5f} ({100 * np.mean(kmap > 1):.0f} % from 4x), "
          f"own-pixel decision {naive_bias:+.5f} ({100 * own.mean():.0f} %)")
    assert abs(bias) < 4e-4 and naive_bias > 2.5 * abs(bias) + 1e-3


def _manifest(path, rows, exposure=True):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "camera_id", "colmap_image_name", "light_id", "path"] + (["exposure_s"] if exposure else []))
        w.writerows(rows)


def test_manifest_exposure_brackets(tmp_path):
    rows = [[0, 1, "s.png", lid, f"{lid}_{e}.npy", e] for e in ("0.5", "2.0", "8.0") for lid in ("ambient", 1, 2)]
    rows.append([0, 1, "s.png", "silhouette", "sil.npy", "0.5"])
    _manifest(tmp_path / "m.csv", rows[::-1])                           # order does not matter
    v = Manifest(tmp_path / "m.csv").views["s.png"]
    assert v["exposure_s"] == 0.5 and v["lights"][1].name == "1_0.5.npy" and v["ambient"].name == "ambient_0.5.npy"
    assert [b["exposure_s"] for b in v["brackets"]] == [2.0, 8.0] and v["brackets"][1]["lights"][2].name == "2_8.0.npy"
    assert v["silhouette"].name == "sil.npy"
    _manifest(tmp_path / "plain.csv", [r[:5] for r in rows[:3]], exposure=False)
    p = Manifest(tmp_path / "plain.csv").views["s.png"]
    assert p["brackets"] == [] and p["exposure_s"] is None and sorted(p["lights"]) == [1, 2]
    _manifest(tmp_path / "short.csv", rows[:5])                         # the 2 s bracket lacks light 2
    with pytest.raises(ValueError, match="repeat the base frames"):
        Manifest(tmp_path / "short.csv")
    _manifest(tmp_path / "mixed.csv", rows[:3] + [r[:5] + [""] for r in rows[3:6]])
    with pytest.raises(ValueError, match="some frames and not others"):
        Manifest(tmp_path / "mixed.csv")


def _measure(d):
    fl = stage6_floor.run(d / "sparse/0", d / "mesh.ply", d / "garment_faces.npy", d / "manifest.csv",
                          d / "lights.json", d / "run", "fabric")
    e = []
    for f in sorted((d / "gt").glob("*.npz")):
        g, p = np.load(f), np.load(d / "run/stage4" / f.name)
        sel = g["unshadowed"] & p["ok"]
        e.append(angle_deg(p["n_ps"][sel], g["n_gt_cam"][sel]))
    return fl, float(np.median(np.concatenate(e)))


@pytest.fixture(scope="module")
def boards(tmp_path_factory):
    """The fabric board at the suite's noise: bright (the texture as is, mean albedo 0.62) and
    dark (x0.05, mean 0.031), each at one exposure and bracketed at 4x and 16x."""
    root, out = tmp_path_factory.mktemp("dark"), {}
    for name, a in (("bright", 1.0), ("dark", 0.05)):
        for ex in ((1.0,), (1.0, 4.0, 16.0)):
            d = root / f"{name}_{len(ex)}"
            make_dataset(d, "board", None, noise=True, seed=3, albedo_scale=a, exposures=ex)
            out[name, len(ex) > 1] = _measure(d)
    return out


def test_a_dark_board_needs_brackets(boards):
    for (name, br), (fl, px) in sorted(boards.items()):
        print(f"\n[D] {name:6s} {'bracketed' if br else 'one exposure'}: PS error median {px:.2f} deg, "
              f"floor {[round(fl[s], 3) for s in (5, 10, 20, 50)]} deg at s = 5/10/20/50 mm", end="")
    (fl1, px1), (flb, pxb) = boards["dark", False], boards["dark", True]
    assert fl1[5] > 3.0                                                 # rule 1: INCONCLUSIVE at one exposure
    assert flb[5] < 0.3 and pxb < 1.25 * boards["bright", False][1]     # bracketed: like a bright board


def test_brackets_never_hurt_a_bright_board(boards):
    """Where most pixels already sit high, a bracket still lowers noise and must not add bias:
    deciding from each pixel's own base value raised the 50 mm floor from 0.022 to 0.038 deg here."""
    (fl1, px1), (flb, pxb) = boards["bright", False], boards["bright", True]
    assert pxb <= px1 and all(flb[s] <= fl1[s] * 1.02 for s in (5, 10, 20, 50))


@pytest.mark.parametrize("albedo", [0.15, 1.0])
def test_stage2_checks_the_bracket_frames(tmp_path, albedo):
    """A dark torso (x0.15) and the bright one, whose 4x frames clip over most of the garment."""
    make_dataset(tmp_path / "ds", "torso", None, noise=True, seed=4, angles=[0], albedo_scale=albedo, exposures=(1.0, 4.0))
    stage1_gw_prep.run(tmp_path / "ds", tmp_path / "ds/manifest.csv", tmp_path / "gw")
    v = Manifest(tmp_path / "gw/manifest.csv").views["step0000.png"]
    assert [b["exposure_s"] for b in v["brackets"]] == [4.0]           # stage 1 kept the column
    r = stage2_frame_gate.run(tmp_path / "gw", tmp_path / "ds/mesh.ply", tmp_path / "s2", manifest=tmp_path / "gw/manifest.csv")
    al = r["views"]["step0000.png"]["alignment"]
    br = al["brackets"][0]
    worst = max(np.hypot(*x) for x in br["per_light_px"].values())
    print(f"\n[D] stage 2, albedo x{albedo}: base set {np.hypot(*al['set_shift_px']):.3f} px, 4x set against "
          f"the base {np.hypot(*br['set_shift_px']):.3f} px, worst 4x frame against its base frame {worst:.3f} px")
    assert al["pass"] and al["brackets"][0]["pass"]
    p = v["brackets"][0]["lights"][5]                                   # one bracket frame bumped by 1 px
    np.save(p, np.roll(np.load(p), 1, axis=1))
    r = stage2_frame_gate.run(tmp_path / "gw", tmp_path / "ds/mesh.ply", tmp_path / "s2b", manifest=tmp_path / "gw/manifest.csv")
    al = r["views"]["step0000.png"]["alignment"]
    assert not r["pass"] and al["brackets"][0]["failing_lights"] == ["5"] and al["failing_lights"] == []
    assert json.loads((tmp_path / "s2b/stage2.json").read_text())["pass"] is False
