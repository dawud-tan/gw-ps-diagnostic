"""2. No false alarms: mesh == ground truth. Without noise every eligible face has
theta_s < 0.3 deg at every scale; with noise the verdict rule finds no cluster."""
import numpy as np

from gwps.compare import SCALES_MM


def test_no_false_alarm_noise_free(synth):
    ds = synth.dataset("torso", None, "none")
    _, res, _ = synth.stage5(ds)
    garment = np.load(ds / "garment_faces.npy")
    for s in SCALES_MM:
        th = res[f"theta_{s}"][garment & (res[f"n_views_{s}"] >= 2)]
        th = th[np.isfinite(th)]
        print(f"\n[test2 none] s={s} mm faces={th.size} median={np.median(th):.5f} max={th.max():.5f} deg")
        assert th.size > 10000
        assert th.max() < 0.3


def test_no_false_alarm_noisy(synth):
    _, run, res = synth.verdict(None, "noisy")
    print("\n" + (run / "verdict.md").read_text())
    for s in SCALES_MM:
        assert res["clusters"][s] == [], f"false cluster at s={s} mm"
    assert res["verdict"] == "BAKE"
