"""C3. Ball centres from backlit silhouettes, with a stand attached below the ball."""
import numpy as np


def test_ball_centres(pilot):
    d = pilot.data()
    _, rep = pilot.lights(d)
    t = pilot.truth(d)
    worst, worst_lat = 0.0, 0.0
    for pos, c in rep["ball_centres_m"].items():
        ct = np.array(t["matte_centres_m"][pos])
        e = np.array(c) - ct
        lat = np.linalg.norm(e - (e @ ct) * ct / (ct @ ct))
        st = rep["silhouette"][pos]
        print(f"\n[C3] {pos}: err {np.round(e * 1000, 3)} mm, lateral {lat * 1e6:.1f} um, "
              f"{st['inliers']}/{st['n_edges']} edge points, rms {st['rms_px']:.3f} px", end="")
        worst, worst_lat = max(worst, np.linalg.norm(e)), max(worst_lat, lat)
    assert worst < 0.2e-3
    assert worst_lat < 20e-6       # lateral error is what moves the mirror-ball rays
