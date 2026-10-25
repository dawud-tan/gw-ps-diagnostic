"""B. The board generator, bikin_papan_charuco_gwps.py in github.com/dawud-tan/charuco (moved there on
2026-10-25): the drawing is OpenCV's board definition, the PDF prints at exact scale, the design follows
the camera, and the pipeline's own renderer and detector find the board described by the board.json it
writes. Skipped without a checkout of that repository: clone it next to this one (../charuco), or set
CHARUCO_DIR."""
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from gwps.charuco import BoardConfig

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = Path(os.environ.get("CHARUCO_DIR", ROOT.parent / "charuco")) / "bikin_papan_charuco_gwps.py"
pytestmark = pytest.mark.skipif(not GENERATOR.exists(), reason=f"{GENERATOR} not found: clone "
                                "git@github.com:dawud-tan/charuco.git next to this repository or set CHARUCO_DIR")
if GENERATOR.exists():
    _spec = importlib.util.spec_from_file_location("bikin_papan_charuco_gwps", GENERATOR)
    bpc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(bpc)
CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)


def _corners_vs_definition(img, cfg, px_per_mm):
    """Detect, refine, fit a similarity: -> ids, scale (px/mm), residual rms (px)."""
    cc, ci, _, _ = cv2.aruco.CharucoDetector(cfg.board()).detectBoard(img)
    ids = ci.ravel()
    keep = cfg.usable_corners()[ids]
    uv = cv2.cornerSubPix(img.astype(np.float32), cc[keep].copy(), (5, 5), (-1, -1), CRIT).reshape(-1, 2) + 0.5
    obj = cfg.corners()[ids[keep]][:, :2] * 1000.0                      # board mm
    A = np.c_[obj, np.ones(len(obj))]
    cx, cy = (np.linalg.lstsq(A, uv[:, k], rcond=None)[0] for k in (0, 1))
    res = uv - np.c_[A @ cx, A @ cy]
    return ids[keep], (cx[0] + cy[1]) / 2, float(np.sqrt((res ** 2).sum(1).mean()))


def _mid_design(tmp_path):
    """The generator's design and its own board config, and the pipeline's BoardConfig from its board.json."""
    d = bpc.rancangPapan(3000, 1920, 1440, 1.45, (420.0, 297.0), 10.0, 2, "fabric")
    own = bpc.konfigurasiPapan(d, 0.001)
    own.simpan(tmp_path / "board.json")
    return d, own, BoardConfig.load(tmp_path / "board.json")


def test_drawing_matches_board_definition(tmp_path):
    d, own, cfg = _mid_design(tmp_path)
    img = bpc.rasterkanPapan(bpc.daftarPersegiPapan(own, "fabric"), d["board_mm"], 10.0)
    ids, scale, rms = _corners_vs_definition(img, cfg, 10.0)
    print(f"\n[B] raster at 10 px/mm: {len(ids)} usable corners of {int(cfg.usable_corners().sum())}, "
          f"scale {scale:.4f} px/mm, residual rms {rms:.3f} px")
    assert set(ids) == set(np.flatnonzero(cfg.usable_corners()))
    assert abs(scale / 10 - 1) < 2e-4 and rms < 0.1
    x0, y0, x1, y1 = (np.array(cfg.fabric_region_m) * 1000 * 10).astype(int)
    assert img[y0 + 10:y1 - 10, x0 + 10:x1 - 10].min() == 255            # blank fabric window


@pytest.mark.skipif(shutil.which("pdftoppm") is None, reason="pdftoppm not installed")
def test_pdf_prints_at_exact_scale(tmp_path):
    """Rasterise the PDF at 254 and 508 dpi. A rasteriser snaps rectangle edges to whole pixels
    (~0.29 px rms), so an exact PDF shows a constant pixel scatter that halves in mm when the
    dpi doubles; a geometric error would not shrink."""
    out = tmp_path / "board"
    r = subprocess.run([sys.executable, str(GENERATOR), "--fx-px", "3000", "--width-px", "1920",
                        "--height-px", "1440", "--paper", "A3", "--no-self-check", "--out", str(out)],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    pdf = next(out.glob("*.pdf"))
    cfg = BoardConfig.load(out / "board.json")
    rms_mm = {}
    for dpi in (254, 508):
        subprocess.run(["pdftoppm", "-r", str(dpi), "-gray", "-png", str(pdf), str(tmp_path / f"r{dpi}")], check=True)
        img = cv2.imread(str(next(tmp_path.glob(f"r{dpi}*.png"))), cv2.IMREAD_GRAYSCALE)
        ppm = dpi / 25.4
        ids, scale, rms = _corners_vs_definition(img, cfg, ppm)
        page = (img.shape[1] / ppm, img.shape[0] / ppm)
        rms_mm[dpi] = rms / ppm
        print(f"\n[B] PDF at {dpi} dpi: page {page[0]:.1f} x {page[1]:.1f} mm, scale error {100 * (scale / ppm - 1):+.4f} %, "
              f"corner scatter {rms:.3f} px = {rms_mm[dpi] * 1000:.1f} um, {len(ids)} corners", end="")
        assert abs(page[0] - 420) < 0.3 and abs(page[1] - 297) < 0.3
        assert abs(scale / ppm - 1) < 5e-4
    assert rms_mm[508] < 0.65 * rms_mm[254] and rms_mm[508] < 0.02


def test_design_follows_the_camera():
    cases = {"coarse 640x480 on A3": (1000, 640, 480, (420.0, 297.0)),
             "1920x1440 f=3000 on A3": (3000, 1920, 1440, (420.0, 297.0)),
             "24 MP 50 mm on A3": (50 / 0.00597, 6000, 4000, (420.0, 297.0)),
             "coarse 640x480 on A1": (1000, 640, 480, (841.0, 594.0))}
    res = {}
    for name, (fx, W, H, paper) in cases.items():
        d = bpc.rancangPapan(fx, W, H, 1.45, paper, 10.0, 2, "fabric")
        ok = all(r[-1] for r in bpc.periksaRancangan(d, bpc.konfigurasiPapan(d, 0.001)))
        res[name] = (d, ok)
        print(f"\n[B] {name}: {d['squares_x']}x{d['squares_y']} x {d['square_mm']} mm, {d['dictionary']}, "
              f"cell at 50 deg {d['cell_px_at_max_tilt']:.1f} px, checks {'ok' if ok else 'FAIL'}", end="")
    assert not res["coarse 640x480 on A3"][1] and res["coarse 640x480 on A1"][1]
    assert res["1920x1440 f=3000 on A3"][1] and res["1920x1440 f=3000 on A3"][0]["dictionary"].startswith("DICT_5X5")
    hi = res["24 MP 50 mm on A3"][0]
    assert res["24 MP 50 mm on A3"][1] and hi["square_mm"] >= bpc.sisiKotakMinimumMm \
        and hi["n_markers"] <= bpc.jumlahPenandaMaksimum


def test_pipeline_detects_the_board(tmp_path):
    """The generator's board.json, rendered and detected by the pipeline's own code at the checklist
    placements; the generator's standalone self-check must report the same."""
    from gwps.calib_synth import _Rx, _Ry, board_albedo_texture, render_board
    from gwps.camera import Camera
    from gwps.charuco import board_pose, detect, face_normal_cam
    _, own, cfg = _mid_design(tmp_path)
    fx, W, H, distance_m = 3000.0, 1920, 1440, 1.45
    cam = Camera(1, "PINHOLE", W, H, fx, fx, W / 2, H / 2)
    tex, ppm = board_albedo_texture(cfg, px_per_m=max(4000.0, 3 * fx / distance_m))
    rng = np.random.default_rng(0)
    centre_b = np.array([cfg.size_m[0] / 2, cfg.size_m[1] / 2, 0])
    usable = int(cfg.usable_corners().sum())
    own_rows = bpc.periksaSendiri(own, fx, W, H, distance_m)
    print("")
    for (name, ty, tx), (own_name, own_n, own_rms, own_err) in zip(
            (("face-on, 20 deg about horizontal", 0, 20), ("+30 deg", 30, 4), ("-50 deg", -50, 4), ("+50 deg", 50, 4)),
            own_rows):
        R = _Ry(ty) @ _Rx(tx)
        img = render_board(cam, R, np.array([0, 0, distance_m]) - R @ centre_b, tex, ppm, [], rng, noise=True,
                           cfg=cfg, uniform=0.5)
        det = detect(img, cfg)
        Re, _, st = board_pose(det, cfg, cam)
        err = np.degrees(np.arccos(np.clip(face_normal_cam(Re) @ face_normal_cam(R), -1, 1)))
        print(f"[B] {name}: {len(det.ids)} corners, rms {st['rms_px']:.3f} px, normal error {err:.3f} deg "
              f"(generator's self-check: {own_n}, {own_rms:.3f} px, {own_err:.3f} deg)")
        assert len(det.ids) == usable and err < 0.1
        assert own_name == name and own_n == len(det.ids) and abs(own_err - err) < 1e-6
