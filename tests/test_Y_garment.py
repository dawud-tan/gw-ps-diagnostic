"""Y. Garment face selection (stage_garment.py) on a synthetic dressed mannequin: the region cut
alone keeps the exposed mannequin; a bare-mannequin reference from another session, once registered,
removes it; per-view garment masks vote faces in; floaters go; the CLI writes what the stages read."""
import cv2
import numpy as np
import pytest

import stage_garment
from gwps import garment as G
from gwps.camera import Camera
from gwps.raycast import Caster, cast_view
from gwps.synth_garment import make_scene
from gwps.synth_sfm import look_at


@pytest.fixture(scope="module")
def scene():
    return make_scene()


def _pr(sel, truth, area):
    """Precision and recall by area."""
    tp = area[sel & truth].sum()
    return tp / area[sel].sum(), tp / area[truth].sum()


def test_region_alone_keeps_the_mannequin(scene):
    scan, truth, _, _ = scene
    sel, rep = G.select(scan)
    p, r = _pr(sel, truth, scan.area_faces)
    print(f"\n[Y] region only: precision {p:.4f}, recall {r:.4f}, cleanup dropped {rep['cleanup']['faces_dropped']} faces")
    c = scan.triangles_center
    assert r > 0.999 and p < 0.9                                  # neck, shoulders, arm and body strips stay in
    assert not sel[np.hypot(c[:, 0], c[:, 1]) > 0.6].any() and not sel[c[:, 2] < 0.01].any()
    assert rep["cleanup"]["faces_dropped"] > 0                     # the small floater near the garment


def test_registered_reference_removes_the_mannequin(scene):
    scan, truth, bare, T = scene
    sel, rep = G.select(scan, reference=bare)
    p, r = _pr(sel, truth, scan.area_faces)
    reg = rep["registration"]
    Rf, tf = np.array(reg["R"]), np.array(reg["t_m"])                 # should undo the reference's move T
    err_deg = np.degrees(np.arccos(np.clip((np.trace(Rf @ T[:3, :3]) - 1) / 2, -1, 1)))
    err_mm = 1000 * np.linalg.norm(Rf @ T[:3, 3] + tf)
    print(f"\n[Y] reference: precision {p:.4f}, recall {r:.4f}; registration {reg['rotation_deg']:.2f} deg "
          f"(applied {np.degrees(np.arccos((np.trace(T[:3, :3]) - 1) / 2)):.2f}), left over {err_deg:.3f} deg and "
          f"{err_mm:.2f} mm, inliers {reg['inlier_fraction_1mm']:.3f}")
    # the rotation is weakly pinned: most exposed parts are round about the axis (turntable top,
    # neck), the elliptic strips move ~0.3 mm per 0.4 deg, and the arm (0.2 m out) is trimmed at the
    # final 1 mm tolerance; 0.4 deg leaves the arm ~1.3 mm off, inside the 2 mm near-test
    assert p > 0.99 and r > 0.99 and err_deg < 0.5 and err_mm < 1.0


def test_unregistered_reference_is_worse(scene):
    scan, truth, bare, _ = scene
    sel, _ = G.select(scan, reference=bare, register=False)
    p, r = _pr(sel, truth, scan.area_faces)
    print(f"\n[Y] reference without registration: precision {p:.4f}, recall {r:.4f}")
    assert p < 0.97


def _views(scan, n=4):
    """Face-id maps of n views around the turntable (640 x 480, f = 1000 px, 1.6 m out)."""
    cam = Camera(1, "PINHOLE", 640, 480, 1000.0, 1000.0, 320.0, 240.0)
    caster = Caster(scan)
    maps = {}
    for k in range(n):
        a = np.radians(360.0 * k / n)
        C = np.array([1.6 * np.sin(a), -1.6 * np.cos(a), 0.35])
        R, t = look_at(C, np.array([0.0, 0.0, 0.35]))
        maps[f"step{k:04d}.png"] = cast_view(caster, cam, R, t)["face_id"]
    return maps


def test_garment_masks_vote(scene):
    scan, truth, _, _ = scene
    maps = _views(scan)
    k = np.ones((5, 5), np.uint8)
    masks = {n: cv2.erode(((f >= 0) & truth[np.maximum(f, 0)]).astype(np.uint8), k) > 0 for n, f in maps.items()}
    sel, rep = G.select(scan, face_id_maps=maps, masks=masks)
    base, _ = G.select(scan)
    g, nlab = G.vote(maps, masks, len(scan.faces))
    lab = nlab > 0
    p_lab, _ = _pr(sel & lab, truth & lab, scan.area_faces)
    p, r = _pr(sel, truth, scan.area_faces)
    p0, _ = _pr(base, truth, scan.area_faces)
    print(f"\n[Y] garment masks (4 views, eroded 2 px): precision on labelled faces {p_lab:.4f}; overall precision "
          f"{p:.4f} (region only {p0:.4f}), recall {r:.4f}; labelled area {rep['masks']['area_labelled_m2']:.3f} m^2")
    assert p_lab > 0.99 and p > p0


def test_cli(scene, tmp_path):
    scan, truth, bare, _ = scene
    scan.export(tmp_path / "scan.ply")
    bare.export(tmp_path / "bare.ply")
    s3 = tmp_path / "stage3"
    s3.mkdir()
    mdir = tmp_path / "masks"
    mdir.mkdir()
    for name, f in _views(scan, 2).items():
        np.savez_compressed(s3 / f"{name[:-4]}.npz", face_id=f)
        cv2.imwrite(str(mdir / name), ((f >= 0) & truth[np.maximum(f, 0)]).astype(np.uint8) * 255)
    sel, rep = stage_garment.run(tmp_path / "scan.ply", tmp_path / "out/garment_faces.npy", tmp_path / "bare.ply",
                                 garment_masks=mdir, stage3=s3)
    saved = np.load(tmp_path / "out/garment_faces.npy")
    assert saved.dtype == bool and len(saved) == len(scan.faces) and np.array_equal(saved, sel)
    assert (tmp_path / "out/garment_faces.ply").exists() and (tmp_path / "out/garment_faces.json").exists()
    p, r = _pr(sel, truth, scan.area_faces)
    assert p > 0.99 and r > 0.99 and rep["masks"]["views"] == 2
