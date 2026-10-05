"""6. Near-light matters: a directional-light solve on near-light data reproduces evidence B
(20.64 deg median, lights 1.5 m, evidence geometry), and the near-light solve removes it."""
import numpy as np

from gwps import ps
from gwps.compare import angle_deg
from gwps.lights import Light


def evidence_b_geometry():
    centre = np.array([0.0, 0.0, 1.5])
    gx, gy, gz = np.meshgrid(np.linspace(-0.25, 0.25, 11), np.linspace(-0.45, 0.45, 19), [-0.1, 0.0, 0.1],
                             indexing="ij")
    X = centre + np.stack([gx, gy, gz], -1).reshape(-1, 3)
    phis = np.deg2rad(np.arange(0, 360, 60))
    off = np.deg2rad(35)
    l = np.stack([np.sin(off) * np.cos(phis), np.sin(off) * np.sin(phis), -np.cos(off) * np.ones(6)], -1)
    N = [np.array([0, 0, -1.0])]
    for t in np.deg2rad([20, 40]):
        for a in np.deg2rad(np.arange(0, 360, 45)):
            N.append(np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), -np.cos(t)]))
    return centre, X, l, np.array(N)


def test_directional_reproduces_evidence_b_and_near_light_removes_it():
    centre, X, l, N = evidence_b_geometry()
    lights = [Light(i, "ps", "point", centre + 1.5 * d) for i, d in enumerate(l)]
    b_near = ps.light_vectors(lights, X)
    b_dir = np.repeat(ps.light_vectors(lights, centre[None]), len(X), axis=1)
    e_dir, e_near = [], []
    quiet = ps.NoiseModel(read=1e-9, full_well=1e12)
    flat = ps.NoiseModel(read=1.0, full_well=1e30)
    as_evidence = ps.PSParams(shadow_frac=0, min_signal_sigma=-1, outlier_iters=0)
    for n in N:
        I = np.clip(np.einsum("kni,i->kn", b_near, n), 0, None)
        sat = np.zeros_like(I, bool)
        # evidence B's estimator: unweighted least squares over all 6 lights, no rejection
        e_dir.append(angle_deg(ps.solve(I, sat, b_dir, flat, as_evidence)["n"], n))
        e_near.append(angle_deg(ps.solve(I, sat, b_near, quiet)["n"], n))
    e_dir, e_near = np.concatenate(e_dir), np.concatenate(e_near)
    print(f"\n[test6 evidence-B] directional median={np.median(e_dir):.2f} p90={np.percentile(e_dir, 90):.2f} "
          f"max={e_dir.max():.2f} deg; near-light median={np.median(e_near):.2e} max={e_near.max():.2e} deg")
    assert abs(np.median(e_dir) - 20.64) < 0.5
    assert e_near.max() < 0.01


def test_directional_bias_on_synthetic_scene(synth):
    ds = synth.dataset("torso", None, "none")
    near, dirn = synth.stages34(ds), synth.stages34(ds, directional=True)
    e_near, e_dir = [], []
    for f in sorted((ds / "gt").glob("*.npz")):
        g = np.load(f)
        sel = g["unshadowed"]
        for run, e in ((near, e_near), (dirn, e_dir)):
            p = np.load(run / "stage4" / f.name)
            s = sel & p["ok"]
            e.append(angle_deg(p["n_ps"][s], g["n_gt_cam"][s]))
    e_near, e_dir = np.concatenate(e_near), np.concatenate(e_dir)
    print(f"[test6 scene] directional median={np.median(e_dir):.2f} p90={np.percentile(e_dir, 90):.2f} deg; "
          f"near-light median={np.median(e_near):.4f} deg")
    assert np.median(e_dir) > 5.0
    assert np.median(e_near) < 0.1
