"""H. The vertical-bias blind spot, checked by board coverage: stage 6 records each board
placement's elevation span in the camera, stage 5 each face's, and stage 7 says which garment
rows no placement covered and what the board showed at a cluster's own height."""
import json

import numpy as np
import pytest

import stage3_mesh_maps
import stage4_ps
import stage5_compare
import stage6_floor
import stage7_verdict
from gwps.synth import board_mesh, make_dataset
from gwps.verdict import decide, row_coverage, verdict_markdown

BOARD_MID = dict(angles=[0, 30, -30, 50, -50])
BOARD_THREE = dict(angles=[0, 30, -30, 50, -50, 0, 0], heights=[0, 0, 0, 0, 0, 0.15, -0.15])


def _grid():
    """A 120 x 120 grid mesh whose faces get elevations -0.2 .. 0.2 by row."""
    mesh, _ = board_mesh()
    z = mesh.triangles_center[:, 2]
    elev = (0.2 * (2 * (z - z.min()) / (z.max() - z.min()) - 1)).astype(np.float32)
    return mesh, elev


def test_row_coverage_reports_the_uncovered_rows():
    mesh, elev = _grid()
    F = len(mesh.faces)
    faces = {"elev": elev, "n_views_any": np.full(F, 4, np.int16)}
    garment = np.ones(F, bool)
    pl = [{"name": "mid", "elev": [-0.05, 0.05]}, {"name": "up", "elev": [-0.2, -0.1]}]
    rc = row_coverage(mesh, faces, garment, pl)
    assert rc["covered_area_fraction"] == pytest.approx(0.5, abs=0.02) and not rc["pass"]
    assert [[round(a, 2), round(b, 2)] for a, b in rc["uncovered_elev"]] == [[-0.1, -0.05], [0.05, 0.2]]
    pl.append({"name": "down", "elev": [0.04, 0.21]})
    pl.append({"name": "gap", "elev": [-0.11, -0.04]})
    rc = row_coverage(mesh, faces, garment, pl)
    assert rc["pass"] and rc["uncovered_elev"] == []
    assert row_coverage(mesh, faces, garment, None) is None                       # older stage 6 output
    assert row_coverage(mesh, {"n_views_any": faces["n_views_any"]}, garment, pl) is None   # older stage 5


def test_a_covered_vertical_cluster_within_its_heights_floor():
    """A vertical, garment-fixed disagreement of 3 deg in the top rows. The pooled floor is low
    (most placements see nothing), so it is a BUILD; but the placement at that height saw 2.8 deg,
    and the verdict says the board there disagreed as much."""
    mesh, elev = _grid()
    F = len(mesh.faces)
    hot = elev < -0.12
    faces = {"elev": elev, "n_views_any": np.full(F, 4, np.int16), "azimuth_span_deg": np.full(F, 90.0, np.float32),
             "mean_view_angle_deg": np.full(F, 20.0, np.float32), "consistency_bin_deg": np.float64(30.0)}
    for s in (5,):
        faces[f"theta_{s}"] = np.where(hot, 3.0, 0.1).astype(np.float32)
        faces[f"n_views_{s}"] = np.full(F, 4, np.int16)
        faces[f"n_bins_{s}"] = np.full(F, 4, np.int16)
        faces[f"mrl_{s}"] = np.ones(F, np.float32)
        faces[f"dmean_{s}"] = np.tile(np.float32([0.05, 0.0, 0.99]), (F, 1))
    pl = [{"name": "mid", "elev": [-0.06, 0.06], "floor_deg": {"5": 0.2}},
          {"name": "up", "elev": [-0.21, -0.06], "floor_deg": {"5": 2.8}},
          {"name": "down", "elev": [0.06, 0.21], "floor_deg": {"5": 0.2}}]
    res = decide(mesh, faces, np.ones(F, bool), {5: 0.2}, (5,), placements=pl)
    (c,) = res["clusters"][5]
    assert res["verdict"] == "BUILD candidate" and c["height_placements"] == ["up"]
    assert c["floor_at_height_deg"] == 2.8 and c["exceeds_floor_at_height"] is False and c["elev_covered_fraction"] == 1.0
    assert res["vertical_warning"] == {"clusters": 1, "of": 1, "all": True, "uncovered": 0, "within_height_floor": 1}
    md = verdict_markdown(res, (5,))
    assert "do not exceed the floor of the board at their own height" in md and "floor at this height 2.80 deg from up" in md
    res = decide(mesh, faces, np.ones(F, bool), {5: 0.2}, (5,), placements=[pl[0], pl[2]])   # nobody up there
    assert res["vertical_warning"]["uncovered"] == 1 and "no board placement covered" in verdict_markdown(res, (5,))


@pytest.fixture(scope="module")
def edge_bias(tmp_path_factory):
    """The torso with a light-fixed vertical bias only in its top and bottom rows (vbias_edge),
    and the fabric board, rendered with the same bias (as a real rig would show it), placed at
    mid height only or at three heights."""
    d = tmp_path_factory.mktemp("rows")
    t = d / "torso"
    make_dataset(t, "torso", "vbias_edge", noise=True, seed=11)
    stage3_mesh_maps.run(t / "sparse/0", t / "mesh.ply", t / "run/stage3", quiet=True)
    stage4_ps.run(t / "sparse/0", t / "manifest.csv", t / "lights.json", t / "run/stage3", t / "run/stage4")
    stage5_compare.run(t / "sparse/0", t / "mesh.ply", t / "garment_faces.npy", t / "run/stage3", t / "run/stage4",
                       t / "run/stage5")
    out = {}
    for name, kw in (("mid", BOARD_MID), ("three", BOARD_THREE)):
        b = d / f"board_{name}"
        make_dataset(b, "board", "vbias_edge", noise=True, seed=12, **kw)
        stage6_floor.run(b / "sparse/0", b / "mesh.ply", b / "garment_faces.npy", b / "manifest.csv", b / "lights.json",
                         b / "run", "fabric")
        fl = json.loads((b / "run/floors.json").read_text())
        out[name] = (fl, stage7_verdict.run(t / "run/stage5", t / "mesh.ply", t / "garment_faces.npy", fl,
                                            d / f"verdict_{name}.md"), (d / f"verdict_{name}.md").read_text())
    return out


def test_a_mid_height_board_leaves_the_bias_unchecked(edge_bias):
    fl, res, md = edge_bias["mid"]
    rc, w = res["row_coverage"], res["vertical_warning"]
    print(f"\n[H] mid-height board: {res['verdict']}, coverage {rc['covered_area_fraction']:.3f}, "
          f"uncovered {np.round(np.degrees(np.arctan(rc['uncovered_elev'])), 1).tolist()} deg, warning {w}")
    assert len(fl["fabric_placements"]) == 5 and all(p["elev"] for p in fl["fabric_placements"])
    assert res["verdict"] == "BUILD candidate" and w["all"] and w["uncovered"] == w["clusters"]
    assert rc["covered_area_fraction"] < 0.6 and not rc["pass"] and len(rc["uncovered_elev"]) == 2
    assert "lie at rows no board placement covered" in md and "below 95 %" in md


def test_boards_at_three_heights_see_the_bias(edge_bias):
    fl, res, md = edge_bias["three"]
    pf = {p["name"]: p["floor_deg"]["5"] for p in fl["fabric_placements"]}
    print(f"\n[H] three heights: {res['verdict']}, coverage {res['row_coverage']['covered_area_fraction']:.3f}, "
          f"placement floors at 5 mm {np.round(list(pf.values()), 2).tolist()} deg")
    assert pf["step0005"] > 3.0 and pf["step0006"] > 3.0 and max(pf[f"step{k:04d}"] for k in range(5)) < 0.3
    assert res["verdict"] == "INCONCLUSIVE" and res["row_coverage"]["covered_area_fraction"] > 0.85
