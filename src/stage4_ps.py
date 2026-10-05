"""Stage 4: near-light PS solve per view, using stage 3 positions for the light vectors.

Writes <out>/<colmap_image_stem>.npz with n_ps (camera frame), albedo, conf, n_lights, ok, low_conf.
--directional reproduces the drafts' distant-light model (for comparison only).
--noise takes the stage-C radiometry.json (measured read noise and full well); without it the
synthetic defaults are used, which do not describe a real sensor. meta.json records each view's
median pooled chi^2_red: far above 1 means the noise or model-error estimate is too small.

Exposure brackets (a dark garment shot at the PS shutter and again at longer ones; manifest
column exposure_s): each light's frames are merged per pixel to the longest exposure that does
not clip (gwps.io.merge_brackets), in units of the PS shutter, and the noise model takes each
pixel's exposure factor. meta.json records the exposures and the fraction of pixels per light
taken from a bracket. Albedo comes out in units of the shortest exposure, like a single one.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gwps import ps
from gwps.camera import load_model
from gwps.io import Manifest, load_view_stack_merged, save_npz, write_meta
from gwps.lights import LightCalibration

MODEL_ERROR = ps.MODEL_ERROR


def run(sparse, manifest, lights, stage3, out, directional=False, read_noise=0.005, full_well=10000.0,
        noise=None, model_error=MODEL_ERROR):
    cams, images = load_model(sparse)
    man = Manifest(manifest)
    cal = LightCalibration.load(lights)
    base = ps.make_noise_model(noise, read_noise, full_well, model_error)
    out = Path(out)
    per_view = {}
    for im in images.values():
        view = man.views[im.name]
        L = cal.ps_lights_cam(im.camera_id, im.R, im.t)
        ids = sorted(L)
        Ls = [L[i] for i in ids]
        I, sat, had_amb, kmap, br = load_view_stack_merged(view, ids)
        m = np.load(Path(stage3) / f"{im.stem}.npz")
        hit = m["hit"]
        X = m["pos_cam"][hit]
        if directional:
            b_ref = ps.light_vectors(Ls, np.median(X, axis=0, keepdims=True).astype(np.float64))
            b_fn = lambda idx: np.repeat(b_ref, len(idx), axis=1)
        else:
            b_fn = lambda idx: ps.light_vectors(Ls, X[idx].astype(np.float64))
        nm = ps.NoiseModel(base.read, base.full_well, had_amb, base.rel)
        Ih, sh = I[:, hit], sat[:, hit]
        kh = None if kmap is None else kmap[:, hit]
        del I, sat, kmap
        r = ps.solve_view(Ih, sh, b_fn, hit, nm, k=kh)
        Hh, Ww = hit.shape
        full = {}
        for k, v in r.items():
            a = np.zeros((Hh, Ww) + v.shape[1:], v.dtype)
            a[hit] = v
            full[k] = a
        conf, med = ps.local_confidence(full["chi2"], full["dof"], full["ok"], full["n_lights"])
        save_npz(out / f"{im.stem}.npz", n_ps=full["n"], albedo=full["albedo"], conf=conf,
                 n_lights=full["n_lights"], ok=full["ok"], low_conf=full["low_conf"])
        per_view[im.name] = {"solved": int(r["ok"].sum()), "low_conf": int(r["low_conf"].sum()),
                             "hits": int(hit.sum()), "ambient_subtracted": had_amb,
                             "median_pooled_chi2_red": med}
        if kh is not None:
            per_view[im.name]["exposures_s"] = br["exposures_s"]
            per_view[im.name]["bracket_fraction"] = {str(k): v for k, v in br["bracket_fraction"].items()}
    write_meta(out, {"sparse": sparse, "manifest": manifest, "lights": lights, "stage3": stage3,
                     "noise": noise or "defaults"},
               {"directional": directional, "noise_model": base.to_dict(), "ps": ps.PSParams().__dict__},
               {"views": per_view})
    return per_view


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("sparse", "manifest", "lights", "stage3", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--directional", action="store_true")
    ap.add_argument("--noise", default=None, help="stage-C radiometry.json (measured noise model)")
    ap.add_argument("--read-noise", type=float, default=0.005, help="used only without --noise")
    ap.add_argument("--full-well", type=float, default=10000.0, help="used only without --noise")
    ap.add_argument("--model-error", type=float, default=MODEL_ERROR)
    a = ap.parse_args()
    print(json.dumps(run(a.sparse, a.manifest, a.lights, a.stage3, a.out, a.directional,
                         a.read_noise, a.full_well, a.noise, a.model_error), indent=1))
