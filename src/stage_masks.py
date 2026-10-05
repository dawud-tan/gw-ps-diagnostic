"""Object masks for a garment session (see gwps/masks.py for why and how).

  stage_masks.py --session captures/shirt01 [--source silhouette|sfm] [--exclude static.png]

--source silhouette (default): from each view's backlit frame (manifest light_id 'silhouette').
--source sfm: from the developed SfM image colmap_images/<name> (matte black backdrop).
--exclude: an 8-bit PNG the size of the images, white where static structure in front of the
  backdrop (the turntable's base, a stand) must never count as object. Paint it once per rig.
Writes <out>/<colmap_image_name> (default out: <session>/masks), 255 = object, and masks.json
with each view's threshold and object fraction. Run after capture_session.py --develop.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from gwps import masks as M
from gwps.gates import linear_from_image
from gwps.io import Manifest, load_linear, write_meta


def run(session, source="silhouette", exclude=None, out=None, dilate_px=None, manifest="manifest.csv"):
    session = Path(session)
    out = Path(out) if out else session / "masks"
    man = Manifest(session / manifest)
    static = M.read_mask(exclude) if exclude else None
    report = {"source": source, "exclude": str(exclude) if exclude else None, "views": {}}
    for name in sorted(man.views):
        colmap_img = session / "colmap_images" / name
        if source == "silhouette":
            src = man.views[name]["silhouette"]
            if src is None:
                raise ValueError(f"{name}: no silhouette frame in {manifest}; use --source sfm or capture one")
            img, _ = load_linear(src)
            mask, info = M.silhouette_mask(img, dilate_px)
        elif source == "sfm":
            img8 = cv2.imread(str(colmap_img), cv2.IMREAD_UNCHANGED)
            if img8 is None:
                raise ValueError(f"{colmap_img}: missing (run capture_session.py --develop first)")
            mask, info = M.sfm_mask(linear_from_image(img8), dilate_px)
        else:
            raise ValueError(f"unknown source {source!r}")
        if colmap_img.exists():
            h, w = cv2.imread(str(colmap_img), cv2.IMREAD_GRAYSCALE).shape
            if (h, w) != mask.shape:
                raise ValueError(f"{name}: mask {mask.shape} does not match the SfM image {(h, w)}")
        mask = M.exclude(mask, static)
        info["object_fraction"] = float(mask.mean())
        M.write_mask(out / name, mask)
        report["views"][name] = info
    fr = [v["object_fraction"] for v in report["views"].values()]
    report["object_fraction"] = {"min": float(min(fr)), "median": float(np.median(fr)), "max": float(max(fr))}
    (out / "masks.json").write_text(json.dumps(report, indent=1))
    write_meta(out, {"session": session, "exclude": exclude}, {"source": source, "dilate_px": dilate_px})
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True)
    ap.add_argument("--source", choices=["silhouette", "sfm"], default="silhouette")
    ap.add_argument("--exclude", default=None, help="static exclusion mask (white = never object)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dilate-px", type=int, default=None, help="default: 0.1 %% of the image width")
    a = ap.parse_args()
    r = run(a.session, a.source, a.exclude, a.out, a.dilate_px)
    print(json.dumps(r["object_fraction"]))
