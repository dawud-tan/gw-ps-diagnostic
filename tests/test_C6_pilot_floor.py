"""C6. The whole pilot chain: calibrated lights + ChArUco board poses + silhouette ball
centres -> stage 6 fabric and sphere floors. Calibration must not raise the fabric floor
materially over the same controls processed with the true lights and poses."""
import numpy as np


def test_pilot_floor(pilot):
    d = pilot.data()
    cal = pilot.controls(d, calibrated=True)
    true = pilot.controls(d, calibrated=False)
    print("")
    for kind in ("fabric", "sphere"):
        for s in (1, 2, 5, 10, 20, 50):
            print(f"[C6] {kind:6s} floor s={s:2d} mm: calibrated {cal[kind][s]:.3f} deg, true lights/poses {true[kind][s]:.3f} deg")
    for s in (5, 10, 20, 50):
        assert cal["fabric"][s] <= true["fabric"][s] + 0.3
        assert cal["fabric"][s] < 1.0
