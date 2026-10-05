"""N. BAKE path: PS normals and albedo baked onto the decimated garment's UV atlas, glTF 2.0
output. Checked against a ridged (corduroy-like) garment whose detail only PS can see."""
import json
import struct

import numpy as np
import pytest
import trimesh

import stage3_mesh_maps
import stage4_ps
import stage_albedo
import stage_bake
from gwps import bake
from gwps.albedo import ColourCorrection, delta_e00
from gwps.compare import angle_deg
from gwps.raycast import Caster
from gwps.synth import CAMERA_MIX, albedo_rgb, make_dataset, ridges_world


@pytest.fixture(scope="module")
def baked(tmp_path_factory):
    d = tmp_path_factory.mktemp("bake")
    ds = d / "ds"
    make_dataset(ds, "torso", "ridges", noise=True, seed=8, rgb=True)
    stage3_mesh_maps.run(ds / "sparse/0", ds / "mesh.ply", d / "s3", quiet=True)
    stage4_ps.run(ds / "sparse/0", ds / "manifest.csv", ds / "lights.json", d / "s3", d / "s4")
    (d / "ccm.json").write_text(json.dumps(ColourCorrection("linear", np.linalg.inv(CAMERA_MIX)).to_json()))
    stage_albedo.apply(ds / "sparse/0", ds / "manifest.csv", ds / "lights.json", d / "s3", d / "s4", d / "ccm.json", d / "alb")
    rep, extra = stage_bake.run(ds / "sparse/0", ds / "mesh.ply", ds / "garment_faces.npy", d / "s3", d / "s4",
                                d / "garment", albedo=d / "alb", size=1024,
                                sheen={"color": [0.3, 0.3, 0.3], "roughness": 0.5})
    return d, rep, extra


def test_baked_normals_carry_the_detail(baked):
    d, rep, (P, n, t, b, ii, jj, nimg) = baked
    gt = trimesh.load(d / "ds/mesh_gt.ply", process=False)
    fid, loc, _ = Caster(gt).first_hit(P + 0.005 * n, -n)
    ok = fid >= 0
    n_true = ridges_world(gt.face_normals[fid[ok]], loc[ok])
    e_bake = angle_deg(bake.decode_normals(nimg[ii, jj][ok], n[ok], t[ok], b[ok]), n_true)
    e_mesh = angle_deg(n[ok], n_true)
    print(f"\n[N] {rep['faces_garment']} -> {rep['faces_decimated']} faces, coverage {rep['texel_coverage_by_views']:.4f}; "
          f"baked normals median {np.median(e_bake):.2f} p90 {np.percentile(e_bake, 90):.2f} deg, "
          f"mesh only median {np.median(e_mesh):.2f} deg")
    assert rep["texel_coverage_by_views"] > 0.99
    assert np.median(e_bake) < 1.0 and np.percentile(e_bake, 90) < 2.0
    assert np.median(e_bake) < 0.25 * np.median(e_mesh)


def test_baked_base_colour(baked):
    d, rep, (P, n, t, b, ii, jj, nimg) = baked
    import cv2
    img = cv2.imread(str(d / "garment_basecolor.png"))[..., ::-1].astype(np.float64) / 255
    lin = np.where(img <= 0.04045, img / 12.92, ((img + 0.055) / 1.055) ** 2.4)
    de = delta_e00(lin[ii, jj], albedo_rgb(P))
    print(f"\n[N] base colour vs true albedo: dE00 median {np.median(de):.2f} p90 {np.percentile(de, 90):.2f}")
    assert np.median(de) < 1.5


def test_glb_container(baked):
    d, rep, _ = baked
    raw = (d / "garment.glb").read_bytes()
    magic, version, length = struct.unpack("<III", raw[:12])
    jlen, jtype = struct.unpack("<II", raw[12:20])
    js = json.loads(raw[20:20 + jlen])
    blen, btype = struct.unpack("<II", raw[20 + jlen:28 + jlen])
    assert (magic, version, length, jtype, btype) == (0x46546C67, 2, len(raw), 0x4E4F534A, 0x004E4942)
    prim = js["meshes"][0]["primitives"][0]
    assert set(prim["attributes"]) == {"POSITION", "NORMAL", "TANGENT", "TEXCOORD_0"}
    assert "min" in js["accessors"][prim["attributes"]["POSITION"]]
    mat = js["materials"][0]
    assert mat["pbrMetallicRoughness"]["metallicFactor"] == 0.0 and "normalTexture" in mat
    assert js["extensionsUsed"] == ["KHR_materials_sheen"] and "KHR_materials_sheen" in mat["extensions"]
    g = trimesh.load(d / "garment.glb")
    geo = list(g.geometry.values())[0]
    assert len(geo.faces) == rep["faces_decimated"]
    assert np.argmax(np.ptp(geo.vertices, 0)) == 1                    # garment height along glTF +Y
    assert rep["glb"]["base_colour"] == "png" and rep["glb"]["within"] and rep["glb"]["bytes"] == len(raw)
    assert [im["mimeType"] for im in js["images"]] == ["image/png", "image/png"]


def _glb_parts(path):
    raw = path.read_bytes()
    jlen = struct.unpack("<I", raw[12:16])[0]
    js = json.loads(raw[20:20 + jlen])
    binary = raw[28 + jlen:]
    imgs = [binary[js["bufferViews"][im["bufferView"]]["byteOffset"]:][:js["bufferViews"][im["bufferView"]]["byteLength"]]
            for im in js["images"]]
    return js, imgs


def test_an_oversized_glb_gets_a_jpeg_base_colour(baked, tmp_path):
    """Over the size budget the base colour (not the normal map) becomes JPEG until it fits."""
    import cv2
    d, rep, _ = baked
    base = cv2.imread(str(d / "garment_basecolor.png"))[..., ::-1].copy()
    nrm = cv2.imread(str(d / "garment_normal.png"))[..., ::-1].copy()
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    UV = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float)
    N = np.tile([0, 0, 1.0], (4, 1))
    T4 = bake.tangents(V, F, UV, N)
    png = bake.write_glb_within(tmp_path / "png.glb", V, N, T4, UV, F, base, nrm, max_mb=100)
    budget = 0.6 * png["bytes"] / 1e6
    r = bake.write_glb_within(tmp_path / "small.glb", V, N, T4, UV, F, base, nrm, max_mb=budget)
    js, (b_img, n_img) = _glb_parts(tmp_path / "small.glb")
    back = cv2.imdecode(np.frombuffer(b_img, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
    lin = lambda x: np.where(x / 255 <= 0.04045, x / 255 / 12.92, ((x / 255 + 0.055) / 1.055) ** 2.4)
    de = delta_e00(lin(back.reshape(-1, 3).astype(float)), lin(base.reshape(-1, 3).astype(float)))
    print(f"\n[N] GLB with PNG textures {png['bytes'] / 1e6:.2f} MB; budget {budget:.2f} MB -> {r['base_colour']}, "
          f"{r['bytes'] / 1e6:.2f} MB; base colour dE00 against the PNG: median {np.median(de):.2f} p99 {np.percentile(de, 99):.2f}")
    assert png["base_colour"] == "png" and r["base_colour"].startswith("jpeg") and r["within"]
    assert [im["mimeType"] for im in js["images"]] == ["image/jpeg", "image/png"]
    assert np.array_equal(cv2.imdecode(np.frombuffer(n_img, np.uint8), cv2.IMREAD_COLOR)[..., ::-1], nrm)
    assert np.median(de) < 1.0 and len(list(trimesh.load(tmp_path / "small.glb").geometry.values())[0].faces) == 2
    r = bake.write_glb_within(tmp_path / "tiny.glb", V, N, T4, UV, F, base, nrm, max_mb=0.01)
    assert r == {**r, "within": False, "base_colour": "jpeg q75"} and (tmp_path / "tiny.glb").exists()


def test_normal_map_convention():
    """+X right (along +u), +Y up (toward decreasing v: UV origin is the image's upper-left)."""
    V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], float)
    F = np.array([[0, 1, 2], [0, 2, 3]])
    UV = np.array([[0, 1], [1, 1], [1, 0], [0, 0]], float)             # world +y is up in the image
    N = np.tile([0, 0, 1.0], (4, 1))
    T4 = bake.tangents(V, F, UV, N)
    assert np.allclose(T4[:, :3], [1, 0, 0]) and np.all(T4[:, 3] == 1)
    n, t = N[:2], T4[:2, :3]
    b = np.cross(n, t) * T4[:2, 3:4]
    up = np.array([[0, 0.3, 1.0], [0.3, 0, 1.0]])
    rgb, _ = bake.encode_normals(up / np.linalg.norm(up, axis=1, keepdims=True), n, t, b)
    assert rgb[0, 1] > 150 and abs(int(rgb[0, 0]) - 128) <= 1                # tilted up -> green
    assert rgb[1, 0] > 150 and abs(int(rgb[1, 1]) - 128) <= 1                # tilted right -> red
