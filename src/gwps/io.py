"""Manifest, linear image loading and per-stage output writing.

Every stage reads images through config/manifest.csv; nothing else parses filenames.
"""
from __future__ import annotations

import csv
import datetime
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

RAW_EXT = {".cr2", ".cr3", ".nef", ".arw", ".dng", ".raf", ".orf", ".rw2", ".pef", ".srw"}
SATURATION = 0.98   # fraction of full scale treated as saturated


class Manifest:
    """views[colmap_image_name] = {"step", "camera_id", "lights": {light_id: path}, "ambient": path,
    "silhouette": path, "exposure_s", "brackets"}. 'silhouette' is the backlit frame (backdrop on,
    PS lights off) that stage_masks.py turns into the view's object mask; it never enters the PS solve.

    Exposure brackets (dark garments): an optional exposure_s column. A view whose ambient and PS
    frames come at several exposures keeps its shortest one (the PS shutter) as "lights"/"ambient",
    so every reader that knows nothing of brackets sees an ordinary view, and lists the longer
    ones in "brackets" = [{"exposure_s", "lights", "ambient"}, ...], ascending. Without the column,
    or with one exposure per view, "brackets" is empty and "exposure_s" is that exposure or None."""

    def __init__(self, path):
        self.path = Path(path)
        root = self.path.parent
        frames = defaultdict(lambda: defaultdict(lambda: {"lights": {}, "ambient": None}))
        self.views = defaultdict(lambda: {"lights": {}, "ambient": None, "silhouette": None,
                                          "exposure_s": None, "brackets": []})
        with open(self.path, newline="") as f:
            for row in csv.DictReader(f):
                name = row["colmap_image_name"]
                v = self.views[name]
                v["step"] = int(row["step"])
                v["camera_id"] = int(row["camera_id"])
                p = Path(row["path"])
                p = p if p.is_absolute() else root / p
                lid = row["light_id"].strip()
                e = (row.get("exposure_s") or "").strip()
                e = float(e) if e else None
                if lid == "silhouette":
                    v[lid] = p
                    continue
                slot = frames[name][e]
                if lid == "ambient":
                    slot["ambient"] = p
                else:
                    if int(lid) in slot["lights"]:
                        raise ValueError(f"{name}: light {lid} listed twice"
                                         + (f" at {e} s" if e is not None else ""))
                    slot["lights"][int(lid)] = p
        for name, by_e in frames.items():
            v = self.views[name]
            if None in by_e and len(by_e) > 1:
                raise ValueError(f"{name}: exposure_s is set for some frames and not others")
            es = sorted(by_e, key=lambda x: -1.0 if x is None else x)
            v["exposure_s"] = es[0]
            v["lights"], v["ambient"] = by_e[es[0]]["lights"], by_e[es[0]]["ambient"]
            for e in es[1:]:
                b = by_e[e]
                if sorted(b["lights"]) != sorted(v["lights"]) or (b["ambient"] is None) != (v["ambient"] is None):
                    raise ValueError(f"{name}: the {e} s bracket must repeat the base frames "
                                     f"(lights {sorted(v['lights'])}, ambient {v['ambient'] is not None})")
                v["brackets"].append({"exposure_s": e, "lights": b["lights"], "ambient": b["ambient"]})
        self.views = dict(self.views)


def load_linear(path):
    """-> (H,W) float32 linear intensity in [0, 1] of full scale, (H,W) bool saturated."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".npy":
        img = np.load(p).astype(np.float32)
    elif ext in RAW_EXT:
        import rawpy
        with rawpy.imread(str(p)) as raw:
            img = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, user_flip=0,
                                  use_camera_wb=False, use_auto_wb=False, user_wb=[1, 1, 1, 1],
                                  output_color=rawpy.ColorSpace.raw).astype(np.float32) / 65535.0
    elif ext in (".tif", ".tiff", ".png"):
        import cv2
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None or img.dtype != np.uint16:
            raise ValueError(f"{p}: only 16-bit linear TIFF/PNG accepted (8-bit needs a response curve)")
        img = img.astype(np.float32) / 65535.0
    else:
        raise ValueError(f"{p}: unsupported image type; camera JPEGs are not linear")
    sat = img >= SATURATION if img.ndim == 2 else (img >= SATURATION).any(-1)
    if img.ndim == 3:
        img = img.mean(-1)
    return img, sat


def load_view_stack(view, light_ids):
    """Ambient-subtracted stack (K,H,W) in light_ids order, and (K,H,W) saturation mask."""
    amb = None
    if view["ambient"] is not None:
        amb, _ = load_linear(view["ambient"])
    I, S = [], []
    for lid in light_ids:
        if lid not in view["lights"]:
            raise KeyError(f"no image for PS light {lid} in manifest")
        img, sat = load_linear(view["lights"][lid])
        I.append(img - amb if amb is not None else img)
        S.append(sat)
    return np.stack(I), np.stack(S), amb is not None


# A bracket frame is used where the next-shorter frame predicts it stays below this fraction of
# full scale, well clear of SATURATION (see merge_brackets).
BRACKET_MARGIN = 0.9


def _local_peak(img, sigma_px=2.0, size=5):
    """Smoothed local maximum: a Gaussian (sigma_px) then a size x size max filter. The pixel's
    own noise is ~4 % of it (1 / (2 pi sigma^2)), and the max keeps it conservative where fine
    texture puts a pixel above its smoothed neighbourhood."""
    import cv2
    k = int(2 * np.ceil(3 * sigma_px) + 1)
    b = cv2.GaussianBlur(np.asarray(img, np.float32), (k, k), sigma_px, borderType=cv2.BORDER_REPLICATE)
    return cv2.dilate(b, np.ones((size, size), np.uint8), borderType=cv2.BORDER_REPLICATE)


def merge_brackets(base_raw, base_amb, base_sat, brackets, margin=BRACKET_MARGIN):
    """Merge one light's exposure bracket per pixel: the longest exposure that does not clip.

    base_raw, base_sat: (H,W) frame at the base (shortest) exposure and its saturation flags;
    base_amb: the ambient frame at that exposure (or 0.0). brackets: [(k, raw, raw_sat, amb), ...]
    with k = exposure / base exposure > 1 and amb the ambient frame shot at that exposure.
    Returns (I, sat, kmap): I ambient-subtracted in base units ((raw - amb) / k where a bracket
    frame is used), sat = saturated even at the base exposure, and kmap the factor each pixel
    came from (1 = base), which the noise model needs (gwps.ps.NoiseModel).

    Which frame a pixel takes must not depend on that frame's own noise, or it is a selection
    bias (the one stage 4's two-pass light choice removes): deciding from the pixel's own base
    value keeps the base exactly where noise pushed it up, and deciding from a long frame's own
    value keeps it exactly where noise left it unclipped. Measured on a bright synthetic board
    (12 x 30 deg views, test noise), deciding from the base pixel raised the 50 mm floor from
    0.022 to 0.038 deg. So bracket j is used where the next-shorter frame's smoothed local peak
    (_local_peak), scaled by the exposure ratio, stays below `margin`: the pixel's own noise is
    ~4 % of that, and the frame whose value is kept never decides. A bracket frame that clips
    anyway (its own flag; texture above the smoothed peak) is not used.
    """
    I = (base_raw - base_amb).astype(np.float32)
    kmap = np.ones(I.shape, np.float32)
    ref, k_ref = base_raw, 1.0
    for k, raw, raw_sat, amb in sorted(brackets, key=lambda b: b[0]):
        if not k > k_ref:
            raise ValueError(f"bracket factor {k}: brackets must be distinct and longer than the base exposure")
        use = (_local_peak(ref) * np.float32(k / k_ref) < margin) & ~raw_sat
        I = np.where(use, (raw - amb) / np.float32(k), I).astype(np.float32)
        kmap[use] = k
        ref, k_ref = raw, k
    return I, base_sat, kmap


def load_view_stack_merged(view, light_ids):
    """load_view_stack with the view's exposure brackets merged (merge_brackets).

    Returns (I, S, had_amb, kmap, info): kmap is None for a view without brackets, in which case
    I and S are exactly load_view_stack's. info: {"exposures_s", "bracket_fraction"} (the
    fraction of each light's unsaturated pixels taken from a bracket frame)."""
    if not view.get("brackets"):
        I, S, had_amb = load_view_stack(view, light_ids)
        return I, S, had_amb, None, {"exposures_s": [view.get("exposure_s")], "bracket_fraction": {}}
    e0 = view["exposure_s"]
    amb0 = load_linear(view["ambient"])[0] if view["ambient"] is not None else 0.0
    ambs = [load_linear(b["ambient"])[0] if b["ambient"] is not None else 0.0 for b in view["brackets"]]
    I, S, K, frac = [], [], [], {}
    for lid in light_ids:
        if lid not in view["lights"]:
            raise KeyError(f"no image for PS light {lid} in manifest")
        raw0, sat0 = load_linear(view["lights"][lid])
        br = []
        for b, amb in zip(view["brackets"], ambs):
            raw, sat = load_linear(b["lights"][lid])
            br.append((b["exposure_s"] / e0, raw, sat, amb))
        i, s, k = merge_brackets(raw0, amb0, sat0, br)
        I.append(i)
        S.append(s)
        K.append(k)
        frac[lid] = float(np.mean(k[~s] > 1)) if (~s).any() else 0.0
        del br
    return (np.stack(I), np.stack(S), view["ambient"] is not None, np.stack(K),
            {"exposures_s": [e0] + [b["exposure_s"] for b in view["brackets"]], "bracket_fraction": frac})


def git_hash():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip()
    except Exception:
        return None


def write_meta(stage_dir, inputs, params, extra=None):
    d = Path(stage_dir)
    d.mkdir(parents=True, exist_ok=True)
    meta = {"inputs": {k: str(v) for k, v in inputs.items()}, "params": params,
            "git_hash": git_hash(), "written": datetime.datetime.now().isoformat(timespec="seconds")}
    if extra:
        meta.update(extra)
    (d / "meta.json").write_text(json.dumps(meta, indent=1, default=float))


def save_npz(path, **arrays):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k: (v.astype(np.float32) if v.dtype == np.float64 else v)
                                 for k, v in arrays.items()})


def load_linear_rgb(path):
    """-> (H,W,3) float32 linear camera RGB in [0, 1] of full scale, (H,W) bool saturated."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".npy":
        img = np.load(p).astype(np.float32)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, -1)
    elif ext in RAW_EXT:
        import rawpy
        with rawpy.imread(str(p)) as raw:
            img = raw.postprocess(gamma=(1, 1), no_auto_bright=True, output_bps=16, user_flip=0,
                                  use_camera_wb=False, use_auto_wb=False, user_wb=[1, 1, 1, 1],
                                  output_color=rawpy.ColorSpace.raw).astype(np.float32) / 65535.0
    elif ext in (".tif", ".tiff", ".png"):
        import cv2
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None or img.dtype != np.uint16:
            raise ValueError(f"{p}: only 16-bit linear TIFF/PNG accepted")
        img = img.astype(np.float32) / 65535.0
        img = np.repeat(img[..., None], 3, -1) if img.ndim == 2 else img[..., ::-1].copy()   # BGR -> RGB
    else:
        raise ValueError(f"{p}: unsupported image type")
    return img, (img >= SATURATION).any(-1)
