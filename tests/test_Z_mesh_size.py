"""Z. GW-like mesh sizes: the torso with the bump at ~2M faces, seen at 1920 x 1440 (faces about the
size of a pixel's footprint), through stages 3-7 and the bake, each in its own process. The bump
must still be found, the bake must hit its face target, and no stage may need more than 6 GB.
The 10M-face, 24 MP run is the same script by hand (bench_mesh_size.py; numbers in CLAUDE.md)."""
import bench_mesh_size


def test_two_million_faces(tmp_path):
    rep = bench_mesh_size.run(tmp_path / "bench", faces=2e6, scale=3.0, views=4)
    print("\n" + (tmp_path / "bench/bench.md").read_text())
    assert rep["faces"] > 1.9e6 and rep["faces_seen_fraction"] > 0.45           # 4 views see about half
    assert rep["verdict"] == "BUILD candidate"
    assert any(c["scale_mm"] == 5 and c["mrl"] is not None and c["mrl"] >= 0.7 for c in rep["bump_clusters"])
    assert rep["bake"]["faces_decimated"] <= 50_000 and rep["bake"]["texel_coverage_by_views"] > 0.45
    assert all(x["peak_rss_gb"] < 6.0 for x in rep["stages"].values())
