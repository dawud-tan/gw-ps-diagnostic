"""I. Implied height error of a cluster, in mm (gwps.height): the low-passed normal difference
integrated over the surface. The integrator on a known surface, then the missing 3 mm bump
(sigma 10 mm): its cluster reports how far outside the mesh the real surface is."""
import numpy as np
import pytest

import stage3_mesh_maps
import stage4_ps
import stage5_compare
import stage7_verdict
from gwps.camera import load_model
from gwps.height import cluster_heights, frankot_chellappa, integrate_masked, runs_ok


def test_runs_ok_needs_the_whole_step_valid():
    s = np.array([[1, 1, 1, 0, 1, 1, 1, 1]], bool)
    assert runs_ok(s, 2, 1).astype(int).tolist() == [[1, 0, 0, 0, 1, 1, 0, 0]]
    assert runs_ok(s.T, 1, 0)[:, 0].astype(int).tolist() == [1, 1, 0, 0, 1, 1, 1, 0]


def test_masked_integration_recovers_a_surface_on_an_irregular_region():
    yy, xx = np.mgrid[:60, :80].astype(float)
    h = 0.01 * xx - 0.004 * yy + 0.5 * np.exp(-((xx - 40) ** 2 + (yy - 30) ** 2) / 50)
    sup = (xx - 40) ** 2 / 35 ** 2 + (yy - 30) ** 2 / 25 ** 2 < 1
    sup[25:35, 5:12] = False                                            # a hole, as a fold's shadow leaves
    du, dv = np.zeros_like(h), np.zeros_like(h)
    du[:, :-1], dv[:-1] = h[:, 1:] - h[:, :-1], h[1:] - h[:-1]
    ok_u, ok_v = np.zeros_like(sup), np.zeros_like(sup)
    ok_u[:, :-1], ok_v[:-1] = sup[:, :-1] & sup[:, 1:], sup[:-1] & sup[1:]
    r = integrate_masked(du, dv, ok_u, ok_v, sup)
    d = (r - h)[sup]
    assert np.abs(d - d.mean()).max() < 1e-5 and np.isnan(r[~sup]).all()
    f = frankot_chellappa(np.where(ok_u, du, 0.0), np.where(ok_v, dv, 0.0))   # zero-filled outside: not h
    e = (f - h)[sup]
    print(f"\n[I] irregular region: masked least squares max error {np.abs(d - d.mean()).max():.1e}, "
          f"Frankot-Chellappa (zero-filled) {np.abs(e - e.mean()).max():.3f} (bump height 0.5)")
    assert np.abs(e - e.mean()).max() > 0.05


@pytest.fixture(scope="module")
def bump(synth):
    """Test 3's scene (noisy, 12 x 30 deg): the mesh lacks a 3 mm bump of sigma 10 mm."""
    ds = synth.dataset("torso", "bump", "noisy")
    run, _, _ = synth.stage5(ds)
    res = stage7_verdict.run(run / "stage5", ds / "mesh.ply", ds / "garment_faces.npy", synth.floors("noisy"),
                             run / "verdict_heights.md", sparse=ds / "sparse/0", stage3=run / "stage3", stage4=run / "stage4")
    return ds, run, res


def test_the_missing_bump_reads_as_its_height(bump):
    ds, run, res = bump
    h5 = [c for c in res["clusters"][5] if c["height_mm"]]
    h10 = [c for c in res["clusters"][10] if c["height_mm"]]
    for s, cl in ((5, h5), (10, h10)):
        for c in cl:
            print(f"\n[I] s = {s} mm cluster ({c['area_mm2']:.0f} mm^2): implied height "
                  f"{c['height_fine_mm']['peak_mm']:+.2f} mm at 2 mm, {c['height_mm']['peak_mm']:+.2f} mm at s, "
                  f"views {c['height_mm']['per_view_peak_mm']}", end="")
    big5, big10 = max(h5, key=lambda c: c["area_mm2"]), max(h10, key=lambda c: c["area_mm2"])
    # the s-low-passed bump's apex is 3 * 100 / (100 + s^2) mm: 2.4 at 5 mm, 1.5 at 10 mm; the
    # reference ring around the cluster still sits on the bump's tail, so a little less is expected
    assert 1.9 < big5["height_mm"]["peak_mm"] < 2.4 and 1.1 < big10["height_mm"]["peak_mm"] < 1.5
    assert 2.4 < big5["height_fine_mm"]["peak_mm"] < 3.0                # at 2 mm: closer to the full 3 mm
    assert big5["height_mm"]["views"] >= 2 and all(c["height_mm"]["peak_mm"] > 0 for c in h5 + h10)
    assert "implied height +" in (run / "verdict_heights.md").read_text()


def test_both_integrators_agree_on_face_on_views(bump):
    """Within 45 deg of face-on the low-passed difference is the gradient of one surface, and the
    masked least squares and Frankot-Chellappa agree; beyond it they did not (3.9 against 2.3 mm
    at 60 deg on this bump), which is why those views are left out."""
    ds, run, res = bump
    c = max(res["clusters"][5], key=lambda c: c["area_mm2"])
    garment = np.load(ds / "garment_faces.npy")
    cams, images = load_model(ds / "sparse/0")
    params = stage7_verdict.stage5_params(run / "stage5")

    def views():
        for im in images.values():
            m = dict(np.load(run / "stage3" / f"{im.stem}.npz"))
            p = np.load(run / "stage4" / f"{im.stem}.npz")
            yield m, {"n": p["n_ps"], "conf": p["conf"], "ok": p["ok"]}, cams[im.camera_id].fx

    lsq = cluster_heights(views(), garment, [("c", c["faces"], 5)], params)["c"]
    fc = cluster_heights(views(), garment, [("c", c["faces"], 5)], params, method="fc")["c"]
    print(f"\n[I] per view: least squares {lsq['per_view_peak_mm']} mm, Frankot-Chellappa {fc['per_view_peak_mm']} mm")
    assert np.allclose(lsq["per_view_peak_mm"], fc["per_view_peak_mm"], atol=0.05)


def test_no_cluster_no_height(synth):
    ds = synth.dataset("torso", None, "noisy")
    run, _, _ = synth.stage5(ds)
    res = stage7_verdict.run(run / "stage5", ds / "mesh.ply", ds / "garment_faces.npy", synth.floors("noisy"),
                             sparse=ds / "sparse/0", stage3=run / "stage3", stage4=run / "stage4")
    assert res["verdict"] == "BAKE" and not any(res["clusters"][s] for s in (5, 10, 20, 50))
