#!/usr/bin/env python3
"""bikin_papan_charuco.py: ChArUco board for the GW x PS rig, sized for the camera that uses it.

Adapted from /WIN_D/protek/charuco/bikin_papan_charuco.py (Dawud Tan). The original made
boards for phone screens, monitors and paper (DICT_4X4_100, 5 x 11 squares, marker 0.7),
recoloured white to grey, and traced the raster into a vector PDF. For this rig:

- **Sized from the camera** (focal length in px, image size, working distance): squares of
  ~60 px at the garment distance but >= 15 mm and <= 250 markers (high-resolution cameras get
  bigger squares, not hundreds of markers); ArUco cells >= 4 px (5x5 dictionary) or >= 3 px
  (4x4) even at 50 deg tilt; the board fits the frame; the fabric window is as large as the
  paper allows. Checks that fail are reported (exit code 1), e.g. a coarse camera on small paper.
- **Fabric window:** a blank (paper-white) centre where the garment fabric is mounted, framed
  by `--frame-squares` squares of ChArUco. Blank rather than printed, so thin fabric shows no
  checker through it. Corners within half a square of the window are ignored by the detector.
- **Exact vector PDF** drawn straight from OpenCV's board definition (every square and every
  marker cell is a rectangle in mm): no raster tracing, no rounding to printer dots.
- **White stays paper white** and black is black (the original's grey "white" lowers contrast;
  fine on a screen, bad for a measurement target). Optional CMYK TIFF with rich black + ICC.
- **board.json** for gwps (`config/board.json`), with the fabric region in board coordinates.
- **Self-check:** renders the board through the camera (with the rig's noise) at the
  recommended placements, detects it with the pipeline's detector, and reports the pose error.

Print the PDF at 100 % ("actual size") on matte paper, glue it flat to a rigid backing, then
**measure a 10-square span with callipers and put the measured square size in board.json**.

Examples:
  bikin_papan_charuco.py --camera config/intrinsics_cam1.json --distance-m 1.45 --paper A3 --out out/board
  bikin_papan_charuco.py --focal-mm 50 --pixel-um 3.76 --width-px 9504 --height-px 6336 --paper A2 --out out/board
"""
import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

PAPER_MM = {"A4": (297.0, 210.0), "A3": (420.0, 297.0), "A2": (594.0, 420.0), "A1": (841.0, 594.0),
            "letter": (279.4, 215.9), "tabloid": (431.8, 279.4)}          # landscape (w, h)
DICTS = {4: [50, 100, 250, 1000], 5: [50, 100, 250, 1000], 6: [50, 100, 250, 1000]}
MARKER_RATIO = 0.72          # marker / square; leaves 14 % of a square between marker and corner
MAX_TILT_DEG = 50.0          # steepest board placement in the capture checklist
TARGET_SQUARE_PX = 60.0      # at the garment distance, face-on
MIN_SQUARE_MM = 15.0         # print accuracy: a printer's absolute error is a bigger fraction of a small square
MAX_MARKERS = 250            # high-resolution cameras get bigger squares, not hundreds of markers
MIN_CELL_PX = {5: 4.0, 4: 3.0}


# ---------------------------------------------------------------- camera
def camera_from_args(a):
    """-> (fx_px, width_px, height_px, description)."""
    if a.camera:
        from gwps.camera import load_camera_model, read_distorted_cameras_txt
        p = Path(a.camera)
        cam = load_camera_model(p) if p.suffix == ".json" else read_distorted_cameras_txt(p)[a.camera_id]
        return float(cam.fx), int(cam.width), int(cam.height), f"{p.name} (camera {cam.camera_id})"
    if a.fx_px:
        fx = a.fx_px
    elif a.focal_mm and a.pixel_um:
        fx = a.focal_mm / (a.pixel_um / 1000.0)
    else:
        raise SystemExit("give --camera, or --fx-px, or --focal-mm with --pixel-um (plus --width-px --height-px)")
    if not (a.width_px and a.height_px):
        raise SystemExit("--width-px and --height-px are required without --camera")
    return float(fx), int(a.width_px), int(a.height_px), f"fx {fx:.0f} px, {a.width_px} x {a.height_px}"


# ---------------------------------------------------------------- design
def design(fx, W, H, distance_m, paper_mm, margin_mm, frame, layout, dictionary=None, square_mm=None):
    """Choose square size, grid and dictionary for this camera and paper. Returns a dict."""
    usable = (paper_mm[0] - 2 * margin_mm, paper_mm[1] - 2 * margin_mm)
    px_mm = fx / (distance_m * 1000.0)                         # image px per mm at the garment distance
    cos_t = math.cos(math.radians(MAX_TILT_DEG))
    min_inner = 3 if layout == "fabric" else 0
    max_sq = min(usable) / (2 * frame + min_inner) if layout == "fabric" else min(usable) / 4
    if square_mm is None:
        # ~TARGET_SQUARE_PX, but never so small that the ArUco cells drop below MIN_CELL_PX at the
        # max tilt (5x5 preferred, 4x4 if the paper is too small), never larger than the paper allows.
        # Squares are in 0.5 mm steps, rounded UP from a minimum and DOWN from the paper limit.
        need = {b: MIN_CELL_PX[b] * (b + 2) / (MARKER_RATIO * px_mm * cos_t) for b in (5, 4)}
        up = lambda v: math.ceil(v * 2 - 1e-9) / 2
        down = lambda v: math.floor(v * 2 + 1e-9) / 2
        target = max(TARGET_SQUARE_PX / px_mm, MIN_SQUARE_MM, math.sqrt(usable[0] * usable[1] / (2 * MAX_MARKERS)))
        sq = None
        for b in (5, 4):
            cand = up(max(min(target, max_sq), need[b]))
            if cand <= max_sq:
                sq = cand
                break
        if sq is None:                                          # camera too coarse for this paper: report
            sq = down(max_sq)
    else:
        sq = float(square_mm)
    sx, sy = int(usable[0] // sq), int(usable[1] // sq)
    marker = round(MARKER_RATIO * sq, 1)
    n_markers = (sx * sy) // 2 + 1
    cell = {b: marker / (b + 2) * px_mm for b in (5, 4)}
    if dictionary:
        name = dictionary
        bits = int(name.split("_")[1][0])
    else:
        bits = 5 if cell[5] * cos_t >= MIN_CELL_PX[5] else 4
        size = next((s for s in DICTS[bits] if s >= n_markers), None)
        if size is None:
            raise SystemExit(f"{n_markers} markers exceed the largest dictionary: use bigger squares (--square-mm)")
        name = f"DICT_{bits}X{bits}_{size}"
    fabric = None
    if layout == "fabric":
        fabric = [frame * sq, frame * sq, (sx - frame) * sq, (sy - frame) * sq]
    board_mm = (sx * sq, sy * sq)
    return {"square_mm": sq, "marker_mm": marker, "squares_x": sx, "squares_y": sy, "dictionary": name,
            "bits": bits, "n_markers": n_markers, "board_mm": board_mm, "fabric_region_mm": fabric,
            "px_per_mm": px_mm, "cell_px_face_on": marker / (bits + 2) * px_mm,
            "cell_px_at_max_tilt": marker / (bits + 2) * px_mm * cos_t, "square_px_face_on": sq * px_mm,
            "board_px": (board_mm[0] * px_mm, board_mm[1] * px_mm), "image_px": (W, H), "paper_mm": paper_mm,
            "margin_mm": margin_mm, "frame_squares": frame, "layout": layout, "distance_m": distance_m}


def board_config(d, thickness_m):
    from gwps.charuco import BoardConfig
    fr = d["fabric_region_mm"]
    return BoardConfig(d["dictionary"], d["squares_x"], d["squares_y"], d["square_mm"] / 1000, d["marker_mm"] / 1000,
                       False, None if fr is None else tuple(x / 1000 for x in fr), thickness_m)


def checks(d, cfg):
    """(name, value, requirement, ok) rows for the design report."""
    W, H = d["image_px"]
    usable = int(cfg.usable_corners().sum())
    rows = [
        ("square, face-on at the garment distance", f"{d['square_px_face_on']:.0f} px", ">= 30 px", d["square_px_face_on"] >= 30),
        (f"ArUco cell at {MAX_TILT_DEG:.0f} deg tilt", f"{d['cell_px_at_max_tilt']:.1f} px",
         f">= {MIN_CELL_PX[d['bits']]:.0f} px ({d['bits']}x{d['bits']})", d["cell_px_at_max_tilt"] >= MIN_CELL_PX[d["bits"]]),
        ("board in the frame at the garment distance", f"{d['board_px'][0]:.0f} x {d['board_px'][1]:.0f} px of {W} x {H}",
         "<= 90 % of each side", d["board_px"][0] <= 0.9 * W and d["board_px"][1] <= 0.9 * H),
        ("board width at 0.8x distance (intrinsics shots)", f"{d['board_px'][0] / 0.8 / W:.0%} of the frame",
         ">= 30 %", d["board_px"][0] / 0.8 / W >= 0.3),
        ("usable corners (outside the fabric window)", f"{usable}", ">= 20", usable >= 20),
        ("dictionary ids", f"{d['n_markers']} markers in {d['dictionary']}", "enough ids", d["n_markers"] <= int(d["dictionary"].split("_")[-1])),
    ]
    if d["fabric_region_mm"]:
        fr = d["fabric_region_mm"]
        short = min(fr[2] - fr[0], fr[3] - fr[1])
        rows.append(("fabric window", f"{fr[2] - fr[0]:.0f} x {fr[3] - fr[1]:.0f} mm", "short side >= 100 mm (150+ better)", short >= 100))
    return rows


# ---------------------------------------------------------------- geometry
def rectangles(cfg, layout):
    """Board drawing as rectangles in board mm (x right, y down): (x, y, w, h, 0=black|1=white).
    Black squares, then each marker as a black square with white cells for its 1-bits."""
    b = cfg.board()
    sq = cfg.square_length_m * 1000
    ml = cfg.marker_length_m * 1000
    d = b.getDictionary()
    nb = d.markerSize
    cell = ml / (nb + 2)
    ids = b.getIds().ravel()
    objp = [np.asarray(o) * 1000 for o in b.getObjPoints()]
    white = {}
    for mid, o in zip(ids, objp):
        c = o[:, :2].mean(0)
        white[(int(c[1] // sq), int(c[0] // sq))] = (int(mid), o[0, :2])
    fr = cfg.fabric_region_m

    def in_window(i, j):
        if layout != "fabric" or fr is None:
            return False
        x0, y0, x1, y1 = (v * 1000 for v in fr)
        return x0 - 1e-6 <= j * sq and (j + 1) * sq <= x1 + 1e-6 and y0 - 1e-6 <= i * sq and (i + 1) * sq <= y1 + 1e-6

    rects = []
    for i in range(cfg.squares_y):
        for j in range(cfg.squares_x):
            if in_window(i, j):
                continue
            if (i, j) not in white:
                rects.append((j * sq, i * sq, sq, sq, 0))
                continue
            mid, tl = white[(i, j)]
            rects.append((tl[0], tl[1], ml, ml, 0))
            bits = cv2.aruco.Dictionary.getBitsFromByteList(d.bytesList[mid:mid + 1], nb)
            for r in range(nb):
                for c in range(nb):
                    if bits[r, c]:
                        rects.append((tl[0] + (c + 1) * cell, tl[1] + (r + 1) * cell, cell, cell, 1))
    return rects


def rasterize(rects, board_mm, px_per_mm, ss=4):
    """Anti-aliased raster (0 black .. 255 white) of the board at px_per_mm."""
    W, H = int(round(board_mm[0] * px_per_mm)), int(round(board_mm[1] * px_per_mm))
    k = px_per_mm * ss
    img = np.full((H * ss, W * ss), 255, np.uint8)
    for x, y, w, h, col in rects:
        c0, c1 = int(round(x * k)), int(round((x + w) * k))
        r0, r1 = int(round(y * k)), int(round((y + h) * k))
        img[r0:r1, c0:c1] = 255 if col else 0
    return cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA) if ss > 1 else img


# ---------------------------------------------------------------- outputs
def write_pdf(path, rects, board_mm, paper_mm, title):
    """Minimal vector PDF: page = paper, board centred, one filled rectangle per element."""
    k = 72.0 / 25.4
    Wp, Hp = paper_mm[0] * k, paper_mm[1] * k
    ox, oy = (paper_mm[0] - board_mm[0]) / 2, (paper_mm[1] - board_mm[1]) / 2
    ops, cur = [], None
    for x, y, w, h, col in rects:
        if col != cur:
            ops.append("1 g" if col else "0 g")
            cur = col
        ops.append(f"{(ox + x) * k:.4f} {Hp - (oy + y + h) * k:.4f} {w * k:.4f} {h * k:.4f} re f")
    # a 100 mm scale bar under the board, to check the print with callipers
    y_bar = oy + board_mm[1] + 4.0
    if y_bar + 2.0 < paper_mm[1]:
        ops += ["0 g", f"{ox * k:.4f} {Hp - (y_bar + 1.0) * k:.4f} {100 * k:.4f} {1.0 * k:.4f} re f"]
    stream = zlib.compress("\n".join(ops).encode())
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {Wp:.4f} {Hp:.4f}] /Contents 4 0 R >>".encode(),
        f"<< /Length {len(stream)} /Filter /FlateDecode >>\nstream\n".encode() + stream + b"\nendstream",
        f"<< /Title ({title}) /Author (Dawud Tan) /Producer (bikin_papan_charuco.py) >>".encode(),
    ]
    out, offsets = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"), []
    for n, o in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n".encode() + o + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{o:010d} 00000 n \n".encode() for o in offsets)
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R /Info 5 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    Path(path).write_bytes(bytes(out))


def write_raster(path, img, board_mm, paper_mm, dpi, cmyk_icc=None):
    """Board centred on the paper at dpi. PNG (grey) or, for .tif, CMYK with rich black + ICC."""
    from PIL import Image
    px_mm = dpi / 25.4
    page = np.full((int(round(paper_mm[1] * px_mm)), int(round(paper_mm[0] * px_mm))), 255, np.uint8)
    y0 = (page.shape[0] - img.shape[0]) // 2
    x0 = (page.shape[1] - img.shape[1]) // 2
    page[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    if str(path).endswith((".tif", ".tiff")):
        a = page.astype(np.float32) / 255.0                     # 1 = paper white, 0 = black
        rich = np.array([0.60, 0.40, 0.40, 1.00])              # rich black (C M Y K), paper white = 0
        cmyk = (np.clip((1 - a)[..., None] * rich, 0, 1) * 255).astype(np.uint8)
        im = Image.fromarray(cmyk, "CMYK")
        kw = {"compression": "tiff_adobe_deflate", "dpi": (dpi, dpi)}
        if cmyk_icc and Path(cmyk_icc).exists():
            kw["icc_profile"] = Path(cmyk_icc).read_bytes()
        im.save(path, **kw)
    else:
        Image.fromarray(page, "L").save(path, dpi=(dpi, dpi))


# ---------------------------------------------------------------- self-check
def self_check(cfg, fx, W, H, distance_m, seed=0):
    """Render the board through a pinhole camera like the rig's (noise as in the synthetic tests)
    at the checklist placements and run the pipeline's detector: corners found and normal error."""
    from gwps.calib_synth import _Rx, _Ry, board_albedo_texture, render_board
    from gwps.camera import Camera
    from gwps.charuco import board_pose, detect, face_normal_cam
    cam = Camera(1, "PINHOLE", W, H, fx, fx, W / 2, H / 2)
    tex, ppm = board_albedo_texture(cfg, px_per_m=max(4000.0, 3 * fx / distance_m))
    rng = np.random.default_rng(seed)
    centre_b = np.array([cfg.size_m[0] / 2, cfg.size_m[1] / 2, 0])
    rows = []
    for name, ty, tx in (("face-on, 20 deg about horizontal", 0, 20), ("+30 deg", 30, 4), ("-50 deg", -50, 4),
                         ("+50 deg", 50, 4)):
        R = _Ry(ty) @ _Rx(tx)
        t = np.array([0, 0, distance_m]) - R @ centre_b
        img = render_board(cam, R, t, tex, ppm, [], rng, noise=True, cfg=cfg, uniform=0.5)
        det = detect(img, cfg)
        if len(det.ids) < 6:
            rows.append((name, 0, float("nan"), float("nan")))
            continue
        Re, _, st = board_pose(det, cfg, cam)
        err = np.degrees(np.arccos(np.clip(face_normal_cam(Re) @ face_normal_cam(R), -1, 1)))
        rows.append((name, len(det.ids), st["rms_px"], err))
    return rows


def report_md(d, cfg, rows, sc, cam_desc):
    L = [f"# ChArUco board for {cam_desc}", "",
         f"- Paper {d['paper_mm'][0]:.0f} x {d['paper_mm'][1]:.0f} mm (landscape), board {d['board_mm'][0]:.1f} x "
         f"{d['board_mm'][1]:.1f} mm centred; garment distance {d['distance_m']} m ({d['px_per_mm']:.2f} px/mm).",
         f"- {d['squares_x']} x {d['squares_y']} squares of {d['square_mm']} mm, markers {d['marker_mm']} mm, "
         f"{d['dictionary']}, layout {d['layout']}" + (f", frame {d['frame_squares']} squares" if d["layout"] == "fabric" else "") + ".",
         "", "| check | value | requirement | |", "|---|---|---|---|"]
    L += [f"| {n} | {v} | {r} | {'ok' if ok else '**FAIL**'} |" for n, v, r, ok in rows]
    if sc:
        L += ["", "Self-check (rendered through the camera with the rig's noise, pipeline detector):", "",
              "| placement | corners | reprojection RMS (px) | normal error (deg) |", "|---|---|---|---|"]
        L += [f"| {n} | {c} | {r:.3f} | {e:.3f} |" for n, c, r, e in sc]
    L += ["", "Print the PDF at 100 % on matte paper; glue flat to a rigid backing; measure a 10-square span "
          "with callipers and put the measured square size (and fabric thickness) in board.json."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("camera (one of)")
    g.add_argument("--camera", help="stage-C intrinsics JSON or a COLMAP cameras.txt")
    g.add_argument("--camera-id", type=int, default=1)
    g.add_argument("--fx-px", type=float)
    g.add_argument("--focal-mm", type=float)
    g.add_argument("--pixel-um", type=float, help="sensor pixel pitch in micrometres")
    g.add_argument("--width-px", type=int)
    g.add_argument("--height-px", type=int)
    ap.add_argument("--distance-m", type=float, default=1.45, help="camera to garment/board distance")
    ap.add_argument("--paper", default="A3", help="A4, A3, A2, A1, letter, tabloid, or WxH in mm (e.g. 500x350)")
    ap.add_argument("--margin-mm", type=float, default=10.0)
    ap.add_argument("--layout", choices=["fabric", "full"], default="fabric")
    ap.add_argument("--frame-squares", type=int, default=2)
    ap.add_argument("--square-mm", type=float, help="override the chosen square size")
    ap.add_argument("--dictionary", help="override, e.g. DICT_5X5_100")
    ap.add_argument("--fabric-thickness-mm", type=float, default=1.0)
    ap.add_argument("--dpi", type=int, default=300, help="for the PNG preview / TIFF")
    ap.add_argument("--tiff", action="store_true", help="also write a CMYK TIFF (rich black, ICC)")
    ap.add_argument("--icc", default="/WIN_D/protek/charuco/icc/PrintWideCMYK.icc")
    ap.add_argument("--no-self-check", action="store_true")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    fx, W, H, cam_desc = camera_from_args(a)
    paper = PAPER_MM.get(a.paper) or tuple(float(v) for v in a.paper.lower().split("x"))
    paper = (max(paper), min(paper))
    d = design(fx, W, H, a.distance_m, paper, a.margin_mm, a.frame_squares, a.layout, a.dictionary, a.square_mm)
    cfg = board_config(d, a.fabric_thickness_mm / 1000)
    rows = checks(d, cfg)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"charuco_{a.layout}_{d['squares_x']}x{d['squares_y']}_{d['square_mm']:g}mm"
    rects = rectangles(cfg, a.layout)
    write_pdf(out / f"{stem}.pdf", rects, d["board_mm"], paper, f"ChArUco {stem}")
    img = rasterize(rects, d["board_mm"], a.dpi / 25.4)
    write_raster(out / f"{stem}.png", img, d["board_mm"], paper, a.dpi)
    if a.tiff:
        write_raster(out / f"{stem}.tif", img, d["board_mm"], paper, a.dpi, a.icc)
    cfg.save(out / "board.json")
    sc = None if a.no_self_check else self_check(cfg, fx, W, H, a.distance_m)
    md = report_md(d, cfg, rows, sc, cam_desc)
    (out / f"{stem}.md").write_text(md)
    (out / f"{stem}.json").write_text(json.dumps({"design": d, "checks": [(n, v, r, bool(ok)) for n, v, r, ok in rows],
                                                  "self_check": sc}, indent=1, default=float))
    print(md)
    print(f"written: {out / stem}.pdf (print at 100 %), .png preview, board.json")
    return 0 if all(ok for *_, ok in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
