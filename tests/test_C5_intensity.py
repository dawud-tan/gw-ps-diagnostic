"""C5. E / LED axis / mu from the matte ball. E within 2 % (CLAUDE.md), judged relative to
the mean over lights (the fit absorbs the primer albedo), and the irradiance field that PS
actually uses within 2 % over the garment volume."""
import json

import numpy as np

from gwps.calib_synth import volume_centre
from gwps.lightfit import irradiance_scale


def test_intensity_fit(pilot):
    d = pilot.data()
    lights_json, rep = pilot.lights(d)
    t = pilot.truth(d)
    est = {l["id"]: l for l in json.loads(lights_json.read_text())["cameras"][0]["lights"]}
    rng = np.random.default_rng(5)
    X = volume_centre() + rng.uniform([-0.2, -0.3, -0.15], [0.2, 0.3, 0.15], (3000, 3))
    ratios, fields = [], []
    for k, tl in t["lights"].items():
        e = est[int(k)]
        ratios.append(e["E"] / tl["E"])
        f_est = irradiance_scale(e["E"], np.array(e["axis"]), e["mu"], np.array(e["position_m"]), X)
        f_true = irradiance_scale(tl["E"], np.array(tl["axis"]), tl["mu"], np.array(tl["position_m"]), X)
        fields.append(f_est / f_true)
    ratios = np.array(ratios)
    rel = ratios / ratios.mean() - 1
    F = np.array(fields) / ratios.mean() - 1
    lopo = max(abs(x) for r in rep["lights"].values() for x in r["lopo_rel"])
    print(f"\n[C5] primer albedo absorbed: E_fit/E_true = {ratios.mean():.4f} (true albedo 0.8)")
    print(f"[C5] relative E error per light (%): {np.round(100 * rel, 2)}")
    print(f"[C5] irradiance field error over the garment volume: max {100 * np.abs(F).max():.2f} %, "
          f"per-light max {np.round(100 * np.abs(F).max(1), 2)} %; leave-one-position-out max {100 * lopo:.2f} %")
    assert np.abs(rel).max() < 0.02
    assert np.abs(F).max() < 0.02
