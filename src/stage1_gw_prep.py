"""Stage 1: prepare the GW dataset. Principal-point gate first; if it fails, crop the
undistorted SfM images and every lit image identically so the principal point is centred
(GW hard-codes it), and update the camera and the 2-D observations.

Inputs: the `colmap image_undistorter` output (images/ + sparse/, text or binary; stage_sfm.py
writes it to <out>/undistorted), the lit manifest undistorted to the same PINHOLE camera
(undistort_lit.py; stage_sfm.py writes <out>/lit/manifest.csv) and, optionally, the object masks
undistorted to that camera (stage_sfm.py: <out>/undistorted/masks). Output directory:
  images/            SfM images for GW (cropped if needed); with masks, RGBA with alpha = mask:
                     GW composites alpha onto its background colour in training and, with
                     --use_valid_mask (on in its radegs script), leaves points outside every mask empty
  sparse/0/          cameras.txt, images.txt (POINTS2D shifted), points3D.txt
  lit/ + manifest.csv  lit images cropped identically (if cropped; else the manifest is copied)
  stage1.json        gate result, crop, and the GW command to run on the RTX machine
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from gwps.camera import load_colmap_dir, write_colmap_txt
from gwps.gates import crop_camera, principal_point_gate


def _crop_file(src, dst, crop):
    img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED) if Path(src).suffix.lower() != ".npy" else np.load(src)
    if img is None:
        raise ValueError(f"cannot read {src}")
    x0, y0, w, h = crop
    out = img[y0:y0 + h, x0:x0 + w]
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    if Path(dst).suffix.lower() == ".npy":
        np.save(dst, out)
    else:
        cv2.imwrite(str(dst), out)


def _rgba(img, mask):
    """BGR(A)/grey 8-bit image + bool mask -> BGRA with alpha = mask (255 = object)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = img[..., :3]
    if img.dtype != np.uint8:
        raise ValueError("masked GW images must be 8-bit (developed SfM images)")
    return np.dstack([img, mask.astype(np.uint8) * 255])


def run(undistorted, lit_manifest, out, resolution="1", masks=None):
    undistorted, out = Path(undistorted), Path(out)
    sp = undistorted / "sparse"                       # image_undistorter writes sparse/; others sparse/0/
    sp = sp / "0" if (sp / "0").exists() else sp
    cams, images, points = load_colmap_dir(sp)
    gates = {cid: principal_point_gate(c) for cid, c in cams.items()}
    new_cams = {cid: (crop_camera(c, gates[cid]["crop"]) if gates[cid]["crop"] else c) for cid, c in cams.items()}
    img_dir = undistorted / "images"
    fractions = {}
    for im in images.values():
        crop = gates[im.camera_id]["crop"]
        src, dst = img_dir / im.name, out / "images" / im.name
        if crop:
            im.xys = im.xys - np.array(crop[:2], float)
        if masks is not None:
            if dst.suffix.lower() != ".png":
                raise ValueError(f"{im.name}: masked images carry alpha, so they must be PNG (develop the SfM images as .png)")
            img = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
            m = cv2.imread(str(Path(masks) / im.name), cv2.IMREAD_GRAYSCALE)
            if img is None or m is None or m.shape != img.shape[:2]:
                raise ValueError(f"{im.name}: image or mask missing, or sizes differ")
            if crop:
                x0, y0, w, h = crop
                img, m = img[y0:y0 + h, x0:x0 + w], m[y0:y0 + h, x0:x0 + w]
            dst.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(dst), _rgba(img, m > 127))
            fractions[im.name] = float((m > 127).mean())
        elif crop:
            _crop_file(src, dst, crop)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
    write_colmap_txt(out / "sparse" / "0", new_cams, images, points)
    rows, fields = [], ["step", "camera_id", "colmap_image_name", "light_id", "path"]
    if lit_manifest is not None:                      # lit images, cropped identically
        lit_manifest = Path(lit_manifest)
        with open(lit_manifest, newline="") as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
    for r in rows:
        crop = gates[int(r["camera_id"])]["crop"]
        p = Path(r["path"])
        src = p if p.is_absolute() else lit_manifest.parent / p
        rel = Path("lit") / (p.name if p.is_absolute() else p)
        if crop:
            _crop_file(src, out / rel, crop)
        else:
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, out / rel)
        r["path"] = str(rel)
    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    res = "" if str(resolution) in ("", "none") else f" -r {resolution}"
    report = {"gates": {str(k): v for k, v in gates.items()},
              "cameras": {str(k): {"width": c.width, "height": c.height, "cx": c.cx, "cy": c.cy} for k, c in new_cams.items()},
              "masks": {"dir": str(masks), "object_fraction": fractions} if masks is not None else None,
              "gw_command": f"python gaussian_wrapping/scripts/train_and_extract_gw_radegs.py -s {out.resolve()} "
                            f"-m <gw_out> --no_postprocess{res}",
              "note": "Record the -r used. Keep the raw mesh (<gw_out>/mesh_..._searched.ply)."
                      + (" Images carry the object mask as alpha (black background in GW)." if masks is not None else "")}
    (out / "stage1.json").write_text(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--undistorted", required=True, help="colmap image_undistorter output directory")
    ap.add_argument("--lit-manifest", required=True, help="manifest of lit images undistorted to the same camera")
    ap.add_argument("--out", required=True)
    ap.add_argument("--resolution", default="1", help="GW -r value to record in the command (1 = full resolution)")
    ap.add_argument("--masks", default=None, help="object masks undistorted to the same camera (stage_sfm.py)")
    a = ap.parse_args()
    r = run(a.undistorted, a.lit_manifest, a.out, a.resolution, a.masks)
    print(json.dumps(r, indent=1, default=float))
