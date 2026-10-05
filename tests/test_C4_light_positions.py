"""C4. Light positions. Two independent estimates:
- mirror: highlights on a mirror ball swapped onto the matte ball's seat, triangulated. Its
  dominant error is seat repeatability: a lateral centre error dC tilts each reflected ray by
  ~2 dC / R, so light error ~ 2 L dC / R (6 mm per 0.1 mm here).
- shading: the light position refit from the matte ball's shading (assumes Lambertian primer;
  exact here because the synthetic primer is Lambertian).
CLAUDE.md target: <= 5-10 mm. What the code guarantees is tested; whether a given seat
repeatability meets the target is a capture requirement and is reported, not asserted."""
import numpy as np
import pytest


def light_errors_mm(pilot, d, source="mirror"):
    _, rep = pilot.lights(d, source)
    t = pilot.truth(d)
    return {int(k): 1000 * np.linalg.norm(np.array(r["position_used_m"]) - np.array(t["lights"][str(k)]["position_m"]))
            for k, r in rep["lights"].items()}, rep


def test_mirror_triangulation_exact_seat(pilot):
    e, _ = light_errors_mm(pilot, pilot.data(0.0, 6, parts=("balls",)))
    x = np.array(list(e.values()))
    print(f"\n[C4] exact seat, 6 positions: mirror errors {np.round(x, 2)} mm")
    assert x.max() <= 5.0


def test_cross_check_tracks_mirror_error(pilot):
    """On a real rig the truth is unknown; 'mirror vs shading' is what reveals a bad seat."""
    e, rep = light_errors_mm(pilot, pilot.data())
    print("")
    for k, r in rep["lights"].items():
        print(f"[C4] swap 20 um, light {k}: true mirror error {e[int(k)]:.2f} mm, mirror-vs-shading {r['mirror_vs_shading_mm']:.2f} mm")
    assert all(abs(e[int(k)] - r["mirror_vs_shading_mm"]) <= 2.0 for k, r in rep["lights"].items())


@pytest.mark.parametrize("swap_mm", [0.02, 0.1])
def test_shading_refit(pilot, swap_mm):
    d = pilot.data(swap_mm, 6, parts=("balls",)) if swap_mm != 0.02 else pilot.data()
    e, _ = light_errors_mm(pilot, d, "shading")
    x = np.array(list(e.values()))
    print(f"\n[C4] swap {swap_mm * 1000:.0f} um: shading-refit errors {np.round(x, 2)} mm")
    assert x.max() <= 5.0


@pytest.mark.parametrize("swap_mm,positions", [(0.0, 6), (0.02, 6), (0.1, 6), (0.02, 3)])
def test_protocol_sensitivity_report(pilot, swap_mm, positions):
    """Reported only: how seat repeatability and the number of positions move the error."""
    d = pilot.data() if (swap_mm, positions) == (0.02, 6) else pilot.data(swap_mm, positions, parts=("balls",))
    x = np.array(list(light_errors_mm(pilot, d)[0].values()))
    xs = np.array(list(light_errors_mm(pilot, d, "shading")[0].values()))
    print(f"\n[C4] swap {swap_mm * 1000:.0f} um, {positions} positions: mirror median {np.median(x):.2f} max {x.max():.2f} mm; "
          f"shading-refit median {np.median(xs):.2f} max {xs.max():.2f} mm")
