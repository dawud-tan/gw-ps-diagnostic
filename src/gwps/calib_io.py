"""Calibration capture manifest (config/calib_manifest.csv).

Columns: target, position, camera_id, light_id, path, and optionally exposure_s.
target is one of intrinsics, mirror_ball, matte_ball, fabric_board (geometry and lights),
dark, flat, sweep, drift (radiometry) or colour_chart. position labels one placement (balls:
the same label for the mirror and the matte ball on the same seat; dark/flat: one frame
each; sweep: one exposure each, with exposure_s set). light_id is a PS light id, 'ambient',
'silhouette' (backlit matte-ball frame), 'all' (any well-lit frame), or a combination such
as '1+2' (those PS lights on together, for the additivity check).
A garment session's own calib_manifest.csv holds its metric_board frames: the ChArUco board
upright on the turntable at several turntable angles (positions m0, m1, ...), for the metric
scale and the turntable axis (stage_sfm.py).
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from .io import load_linear

TARGETS = ("intrinsics", "mirror_ball", "matte_ball", "fabric_board", "dark", "flat", "sweep", "drift",
           "colour_chart", "metric_board")


class CalibManifest:
    def __init__(self, path):
        self.path = Path(path)
        root = self.path.parent
        e = defaultdict(lambda: defaultdict(lambda: {"lights": {}}))
        with open(self.path, newline="") as f:
            for row in csv.DictReader(f):
                t = row["target"].strip()
                if t not in TARGETS:
                    raise ValueError(f"{path}: unknown target {t!r}")
                slot = e[t][row["position"].strip()]
                slot["camera_id"] = int(row["camera_id"])
                if row.get("exposure_s"):
                    slot["exposure_s"] = float(row["exposure_s"])
                p = Path(row["path"])
                p = p if p.is_absolute() else root / p
                lid = row["light_id"].strip()
                if lid in ("ambient", "silhouette", "all"):
                    slot[lid] = p
                elif "+" in lid:
                    slot.setdefault("combos", {})[tuple(sorted(int(x) for x in lid.split("+")))] = p
                else:
                    slot["lights"][int(lid)] = p
        self.entries = {t: dict(v) for t, v in e.items()}

    def positions(self, target, camera_id=None):
        return sorted(k for k, v in self.entries.get(target, {}).items()
                      if camera_id is None or v["camera_id"] == camera_id)

    def slot(self, target, position):
        return self.entries[target][position]


def lit_minus_ambient(slot, light_id):
    """Ambient-subtracted image of one light, and its saturation mask (from the raw frame)."""
    img, sat = load_linear(slot["lights"][light_id])
    if "ambient" in slot:
        img = img - load_linear(slot["ambient"])[0]
    return img, sat


def detection_image(slot):
    """Frame for marker detection: 'all' if captured, else the sum of ambient-subtracted PS frames."""
    if "all" in slot:
        return load_linear(slot["all"])[0]
    amb = load_linear(slot["ambient"])[0] if "ambient" in slot else 0
    return np.sum([load_linear(p)[0] - amb for p in slot["lights"].values()], axis=0)
