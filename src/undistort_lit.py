"""Undistort every image listed in a manifest (config/manifest.csv or config/calib_manifest.csv)
to the PINHOLE camera(s), and write a new manifest pointing at the undistorted images.

--distorted: the distorted model(s): COLMAP's original cameras.txt (SIMPLE_RADIAL, RADIAL,
             OPENCV, FULL_OPENCV) or a stage-C intrinsics JSON. Keyed by camera_id, which must
             equal the manifest's camera_id.
--undistorted: COLMAP's undistorted cameras.txt (garment: `colmap image_undistorter` output),
             or `auto` = what image_undistorter would choose (pilot: no SfM model).
Output: <out>/<same relative image path>.png (16-bit linear grey, saturation = full scale),
<out>/<manifest file name>, <out>/cameras.txt (PINHOLE), <out>/meta.json.
"""
import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from gwps.camera import load_camera_model, read_cameras_txt, read_distorted_cameras_txt, write_colmap_txt
from gwps.io import load_linear, write_meta
from gwps.undistort import colmap_undistorted_camera, source_coverage, undistort_image, undistort_maps

_MAPS = {}


def _init(pairs):
    for cid, (src, dst) in pairs.items():
        _MAPS[cid] = undistort_maps(src, dst)


def _one(job):
    cid, src_path, dst_path = job
    img, sat = load_linear(src_path)
    out = undistort_image(img, _MAPS[cid], sat)
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst_path), np.round(np.clip(out, 0, 1) * 65535).astype(np.uint16))
    return str(dst_path)


def load_distorted(path):
    p = Path(path)
    if p.suffix == ".json":
        c = load_camera_model(p)
        return {c.camera_id: c}
    return read_distorted_cameras_txt(p)


def run(manifest, distorted, undistorted, out, workers=None):
    manifest, out = Path(manifest), Path(out)
    src = load_distorted(distorted)
    if str(undistorted) == "auto":
        dst = {cid: colmap_undistorted_camera(c) for cid, c in src.items()}
    else:
        dst = read_cameras_txt(undistorted)
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if "camera_id" not in fields or "path" not in fields:
        raise ValueError(f"{manifest}: needs camera_id and path columns")
    jobs, used = [], set()
    root = manifest.parent
    for r in rows:
        cid = int(r["camera_id"])
        if cid not in src or cid not in dst:
            raise ValueError(f"camera_id {cid}: no distorted or undistorted model")
        p = Path(r["path"])
        sp = p if p.is_absolute() else root / p
        rel = Path(*p.parts[1:]) if p.is_absolute() else p
        dp = (out / rel).with_suffix(".png")
        jobs.append((cid, str(sp), str(dp)))
        r["path"] = str(dp.relative_to(out))
        used.add(cid)
    pairs = {cid: (src[cid], dst[cid]) for cid in used}
    for cid, (s, d) in pairs.items():
        cov = source_coverage(undistort_maps(s, d), (s.height, s.width))
        if cov.mean() < 0.999:
            print(f"note: camera {cid}: {100 * (1 - cov.mean()):.2f} % of undistorted pixels fall outside the source image")
    workers = workers or min(8, os.cpu_count() or 1)
    if workers > 1:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(pairs,)) as ex:
            list(ex.map(_one, jobs, chunksize=4))
    else:
        _init(pairs)
        for j in jobs:
            _one(j)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / manifest.name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    write_colmap_txt(out, {cid: dst[cid] for cid in used}, {})
    (out / "images.txt").unlink(missing_ok=True)
    (out / "points3D.txt").unlink(missing_ok=True)
    write_meta(out, {"manifest": manifest, "distorted": distorted, "undistorted": undistorted},
               {"interpolation": "bilinear (cv2.remap, 1/32 px)", "saturation": "any saturated source -> 1.0"},
               {"cameras": {cid: {"distorted": {"model": s.colmap_model, "params": s.colmap_params,
                                                "width": s.width, "height": s.height},
                                  "undistorted": {"model": d.model, "params": d.colmap_params(),
                                                  "width": d.width, "height": d.height}}
                            for cid, (s, d) in pairs.items()},
                "n_images": len(jobs)})
    return out / manifest.name, out / "cameras.txt"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--distorted", required=True)
    ap.add_argument("--undistorted", default="auto")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    m, c = run(a.manifest, a.distorted, a.undistorted, a.out, a.workers)
    print(json.dumps({"manifest": str(m), "cameras": str(c)}))
