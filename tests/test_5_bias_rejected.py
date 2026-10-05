"""5. A smooth bias defined in the camera frame (so it rotates with the lights relative to
the garment across steps) fails the consistency check -> UNEXPLAINED, while test 3's bump
passes it. Both at the original 12 x 30 deg turntable steps and at the garment plan's
36 x 10 deg: counting every 10 deg view as an independent sample let this bias through
(MRL 0.711 at s = 50 mm, a false BUILD) until views were grouped into 30 deg bins."""
import numpy as np
import pytest

import stage5_compare
from gwps.compare import CompareParams

BIG = (5, 10, 20, 50)


@pytest.mark.parametrize("steps", [12, 36])
@pytest.mark.parametrize("noise", ["none", "noisy"])
def test_camera_frame_bias_is_unexplained(synth, noise, steps):
    _, run, res = synth.verdict("bias", noise, steps)
    print("\n" + (run / "verdict.md").read_text())
    cl = [c for s in BIG for c in res["clusters"][s]]
    assert cl, "bias produced no cluster at s >= 5 mm, so the consistency check was never exercised"
    print(f"[test5 {noise} {steps} steps] clusters={len(cl)} max MRL={max(c['mrl'] for c in cl):.3f} "
          f"(views/bins median {max(c['views_median'] for c in cl):.0f}/{max(c['bins_median'] for c in cl):.0f})")
    assert all(c["mrl"] < 0.7 for c in cl)
    assert res["verdict"] == "UNEXPLAINED"


@pytest.mark.parametrize("steps", [12, 36])
@pytest.mark.parametrize("noise", ["none", "noisy"])
def test_bump_passes_consistency(synth, noise, steps):
    _, _, res = synth.verdict("bump", noise, steps)
    assert max(c["mrl"] for c in res["clusters"][5] + res["clusters"][10]) >= 0.7


def test_bins_lower_the_bias_score_at_10_deg_steps(synth, tmp_path):
    """Why the bins: scored per view (bin width 0), the same bias at 36 x 10 deg steps comes out
    ~0.1 higher than with 30 deg bins (measured 0.711 against 0.615 at s = 50 mm: a false BUILD
    against a clear rejection). The per-view score sits too close to 0.7 to assert the false
    BUILD itself, so this checks the margin the bins buy."""
    import stage7_verdict
    ds, run, binned = synth.verdict("bias", "noisy", 36)
    stage5_compare.run(ds / "sparse/0", ds / "mesh.ply", ds / "garment_faces.npy", run / "stage3", run / "stage4",
                       tmp_path / "stage5", CompareParams(consistency_bin_deg=0))
    per_view = stage7_verdict.run(tmp_path / "stage5", ds / "mesh.ply", ds / "garment_faces.npy", synth.floors("noisy"))
    top = {name: max(c["mrl"] for c in r["clusters"][50]) for name, r in (("per view", per_view), ("binned", binned))}
    print(f"\n[test5 36 steps, s=50 mm] MRL per view {top['per view']:.3f} ({per_view['verdict']}), "
          f"30 deg bins {top['binned']:.3f} ({binned['verdict']})")
    assert top["per view"] >= top["binned"] + 0.05


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN LIMITATION, not a bug to hide: turntable rotation is about the vertical axis and never "
    "moves a face in image row v, so a light-fixed bias that tilts normals vertically as a function "
    "of v is garment-fixed in every view and the consistency check cannot reject it. "
    "Mitigations need a second viewpoint height or lights at different heights; see report. "
    "The verdict flags it instead (test_vertical_build_is_flagged)."))
def test_vertical_row_dependent_bias_is_not_rejected(synth):
    _, run, res = synth.verdict("vbias", "noisy")
    print("\n" + (run / "verdict.md").read_text())
    assert res["verdict"] == "UNEXPLAINED"


def test_vertical_build_is_flagged(synth):
    """The blind spot cannot be rejected, but it can be flagged: every cluster behind the vertical
    bias's BUILD is mostly vertical (measured 0.98-1.00), the real bump's are not (0.53-0.57)."""
    _, run, res = synth.verdict("vbias", "noisy")
    w = res.get("vertical_warning")
    v = [round(c["vertical_fraction"], 3) for s in BIG for c in res["clusters"][s]]
    print(f"\n[test5 vbias] {res['verdict']}, warning {w}, vertical fractions {v}")
    assert res["verdict"] == "BUILD candidate" and w and w["all"]
    assert "mostly vertical" in (run / "verdict.md").read_text()
    _, _, bump = synth.verdict("bump", "noisy")
    vb = [c["vertical_fraction"] for s in (5, 10) for c in bump["clusters"][s]]
    print(f"[test5 bump] vertical fractions {np.round(vb, 3).tolist()}")
    assert "vertical_warning" not in bump and max(vb) < 0.7
