"""Albedo for the glTF material (after stages 3-4).

  calibrate: ColorChecker mounted in the ChArUco board's window (target colour_chart in the
             calibration manifest; chart.json gives its layout in board mm). Board pose from the
             frame's corners, each patch's camera-RGB albedo from the calibrated shading, then a
             colour correction to linear sRGB (linear 3x3 and root-polynomial), leave-one-out ΔE2000.
  apply:     per view, per-channel albedo from the stage-4 PS normals, colour-corrected ->
             albedo_rgb (linear sRGB) in <out>/<stem>.npz and an sRGB preview PNG.
"""
import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import map_coordinates

from gwps import albedo as alb
from gwps import ps
from gwps.calib_io import CalibManifest, detection_image
from gwps.camera import load_camera_model, load_model
from gwps.charuco import BoardConfig, board_pose, detect, face_normal_cam
from gwps.io import Manifest, load_linear_rgb, save_npz, write_meta
from gwps.lights import LightCalibration

warnings.filterwarnings("ignore", module="colour")


def _sample(img, uv):
    """Bilinear samples (M,3) of an (H,W,3) image at COLMAP-convention (u, v)."""
    rows, cols = uv[:, 1] - 0.5, uv[:, 0] - 0.5
    return np.stack([map_coordinates(img[..., c], [rows, cols], order=1) for c in range(img.shape[-1])], -1)


def calibrate(calib_manifest, board, chart, camera, lights, out, camera_id=1, position="c0", reference=None,
              thickness_m=0.0, grid=7):
    man = CalibManifest(calib_manifest)
    cfg = BoardConfig.load(board)
    layout = alb.ChartLayout.load(chart)
    cam = load_camera_model(camera, camera_id)
    slot = man.slot("colour_chart", position)
    R, t, pose = board_pose(detect(detection_image(slot), cfg), cfg, cam)
    n = face_normal_cam(R)
    L = LightCalibration.load(lights).ps_lights_cam(camera_id, np.eye(3), np.zeros(3))
    amb = load_linear_rgb(slot["ambient"])[0] if "ambient" in slot else 0.0
    ids = [k for k in sorted(slot["lights"]) if k in L]
    frames = [load_linear_rgb(slot["lights"][k]) for k in ids]
    g = (np.arange(grid) + 0.5) / grid - 0.5
    gx, gy = np.meshgrid(g, g)
    measured = []
    for cxy in layout.patch_centres_m():
        pb = np.stack([cxy[0] + gx.ravel() * layout.sample_mm / 1000, cxy[1] + gy.ravel() * layout.sample_mm / 1000,
                       np.full(gx.size, -thickness_m)], -1)
        X = pb @ R.T + t
        uv = cam.project(X)
        I = np.stack([_sample(f[0] - amb, uv) for f in frames])                    # (K, M, 3)
        sat = np.stack([_sample(f[1][..., None].astype(np.float32), uv)[..., 0] > 0 for f in frames])
        b = ps.light_vectors([L[k] for k in ids], X)
        rho = alb.per_channel_albedo(I, b, np.broadcast_to(n, X.shape), sat)
        measured.append(np.median(rho, axis=0))
    measured = np.array(measured)
    ref = np.array(json.loads(Path(reference).read_text())) if reference else alb.reference_linear_srgb()[0]
    fits = {}
    for method in ("linear", "rootpoly"):
        cc = alb.fit_colour_correction(measured, ref, method)
        loo = alb.leave_one_out_de00(measured, ref, method)
        fits[method] = {"correction": cc, "loo_de00_median": float(np.median(loo)), "loo_de00_max": float(loo.max()),
                        "fit_de00_median": float(np.median(alb.delta_e00(cc.apply(measured), ref)))}
    best = min(fits, key=lambda m: fits[m]["loo_de00_median"])
    rep = fits[best]["correction"].to_json({
        "chosen_by": "lower leave-one-out median ΔE2000",
        "fits": {m: {k: v for k, v in f.items() if k != "correction"} | {"matrix": f["correction"].matrix.tolist()}
                 for m, f in fits.items()},
        "measured_camera_rgb": measured.tolist(), "reference_linear_srgb": np.asarray(ref).tolist(),
        "board_pose_rms_px": pose["rms_px"]})
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=1))
    return rep


def apply(sparse, manifest, lights, stage3, stage4, ccm, out):
    cams, images = load_model(sparse)
    man = Manifest(manifest)
    cal = LightCalibration.load(lights)
    cc = alb.ColourCorrection.load(ccm)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    per_view = {}
    for im in images.values():
        view = man.views[im.name]
        L = cal.ps_lights_cam(im.camera_id, im.R, im.t)
        ids = sorted(L)
        m3 = np.load(Path(stage3) / f"{im.stem}.npz")
        p4 = np.load(Path(stage4) / f"{im.stem}.npz")
        mask = m3["hit"] & p4["ok"]
        X = m3["pos_cam"][mask].astype(np.float64)
        n = p4["n_ps"][mask].astype(np.float64)
        # pass 1: shading of every light (float32, K x N), the brightest per pixel
        s = np.stack([np.einsum("ij,ij->i", L[k].light_vector(X)[0], n) for k in ids]).astype(np.float32)
        smax = s.max(0)
        amb = load_linear_rgb(view["ambient"])[0][mask] if view["ambient"] is not None else 0.0
        num = np.zeros((len(X), 3))
        den = np.zeros(len(X))
        # pass 2: one light image at a time
        for j, k in enumerate(ids):
            img, sat = load_linear_rgb(view["lights"][k])
            use = (s[j] > 0) & (s[j] > 0.1 * smax) & ~sat[mask]
            w = np.where(use, s[j], 0.0)
            num += w[:, None] * (img[mask] - amb)
            den += w * s[j]
        rho = np.where((den > 0)[:, None], num / np.maximum(den, 1e-12)[:, None], 0.0)
        lin = np.clip(cc.apply(rho), 0, 1)
        img = np.zeros(mask.shape + (3,), np.float32)
        img[mask] = lin
        save_npz(out / f"{im.stem}.npz", albedo_rgb=img, valid=mask)
        cv2.imwrite(str(out / f"{im.stem}_albedo_srgb.png"), alb.srgb8(img)[..., ::-1])
        per_view[im.name] = {"pixels": int(mask.sum())}
    write_meta(out, {"sparse": sparse, "manifest": manifest, "lights": lights, "stage3": stage3, "stage4": stage4,
                     "ccm": ccm}, {"method": cc.method}, {"views": per_view})
    return per_view


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("calibrate")
    for k in ("calib-manifest", "board", "chart", "camera", "lights", "out"):
        c.add_argument(f"--{k}", required=True)
    c.add_argument("--camera-id", type=int, default=1)
    c.add_argument("--position", default="c0")
    c.add_argument("--reference", default=None, help="JSON 24x3 linear-sRGB reference (default: colour-science data)")
    a_ = sub.add_parser("apply")
    for k in ("sparse", "manifest", "lights", "stage3", "stage4", "ccm", "out"):
        a_.add_argument(f"--{k}", required=True)
    a = ap.parse_args()
    if a.cmd == "calibrate":
        r = calibrate(a.calib_manifest, a.board, a.chart, a.camera, a.lights, a.out, a.camera_id, a.position, a.reference)
        print(json.dumps({"method": r["method"], "fits": {m: {k: v for k, v in f.items() if k != "matrix"}
                                                           for m, f in r["fits"].items()}}, indent=1))
    else:
        print(json.dumps(apply(a.sparse, a.manifest, a.lights, a.stage3, a.stage4, a.ccm, a.out), indent=1))
