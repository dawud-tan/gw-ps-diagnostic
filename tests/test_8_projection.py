"""8. Projection matches pycolmap.Camera.img_from_cam within 0.01 px (PINHOLE, off-centre
principal point); rays through projected vertex pixels hit within 0.1 mm of the visible
vertex; pixel-centre rays land back on their pixel centre."""
import numpy as np
import pycolmap
import trimesh

from gwps.camera import load_model
from gwps.raycast import Caster, cast_view


def test_projection_matches_pycolmap(synth):
    ds = synth.dataset("torso", None, "none")
    cams, images = load_model(ds / "sparse/0")
    cam = cams[1]
    assert abs(cam.cx - cam.width / 2) > 5          # off-centre on purpose
    pc = pycolmap.Camera(model="PINHOLE", width=cam.width, height=cam.height,
                         params=[cam.fx, cam.fy, cam.cx, cam.cy])
    mesh = trimesh.load(ds / "mesh.ply", process=False)
    for im in list(images.values())[:3]:
        Xc = im.world_to_cam(mesh.vertices)
        front = Xc[:, 2] > 0.1
        d = np.abs(cam.project(Xc[front]) - pc.img_from_cam(Xc[front]))
        print(f"\n[test8] {im.name} max |ours - pycolmap| = {d.max():.2e} px")
        assert d.max() < 0.01


def test_rays_hit_visible_vertices(synth):
    ds = synth.dataset("torso", None, "none")
    cams, images = load_model(ds / "sparse/0")
    cam = cams[1]
    mesh = trimesh.load(ds / "mesh.ply", process=False)
    caster = Caster(mesh)
    worst = 0.0
    for im in list(images.values())[:3]:
        V = mesh.vertices
        C = im.centre
        # visible = the first hit toward the vertex is (essentially) the vertex itself
        dvec = V - C
        dist = np.linalg.norm(dvec, axis=1)
        _, _, hd = caster.first_hit(C, dvec / dist[:, None])
        uv = cam.project(im.world_to_cam(V))
        inside = (uv[:, 0] > 1) & (uv[:, 0] < cam.width - 1) & (uv[:, 1] > 1) & (uv[:, 1] < cam.height - 1)
        vis = inside & (np.abs(hd - dist) < 1e-5)
        assert vis.sum() > 1000
        # now go the other way: project, build a ray from the pixel coordinate, cast
        rays_w = cam.rays(uv[vis]) @ im.R
        fid, loc, _ = caster.first_hit(C, rays_w)
        err = np.linalg.norm(loc - V[vis], axis=1)
        worst = max(worst, err.max())
        # pixel centres: the hit of pixel (i, j) projects back to (j + 0.5, i + 0.5)
        maps = cast_view(caster, cam, im.R, im.t)
        i, j = np.nonzero(maps["hit"])
        back = cam.project(maps["pos_cam"][i, j])
        np.testing.assert_allclose(back, np.stack([j + 0.5, i + 0.5], -1), atol=1e-3)
    print(f"\n[test8] worst vertex re-hit error = {worst * 1000:.2e} mm")
    assert worst < 1e-4
