"""Stage C: radiometry of the real camera and lights -> radiometry.json.

Reads targets dark, flat, sweep (with exposure_s), drift, and the matte ball at the drift
position (single-light frames plus light combinations such as '1+2') from the calibration
manifest. radiometry.json's "noise" block is what
stage 4 / stage C / stage 6 take with --noise. Gates (starting points, as in CLAUDE.md):
linear-fit R^2 > 0.999 and < 1 % deviation below saturation; additivity 1 +- 1 %;
drift 1 +- 1 %.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from gwps import radiometry
from gwps.calib_io import CalibManifest
import datetime

from gwps.io import git_hash, load_linear

GATES = {"linearity_r2": 0.999, "linearity_dev": 0.01, "additivity": 0.01, "drift": 0.01}


def _stack(man, target, camera_id):
    return np.stack([load_linear(man.slot(target, p)["all"])[0] for p in man.positions(target, camera_id)
                     if "all" in man.slot(target, p)])


def run(calib_manifest, out, camera_id=1, drift_position="p0", drift_light=1):
    man = CalibManifest(calib_manifest)
    rep = {"gates": GATES, "checks": {}}
    ok = True
    if man.positions("dark", camera_id) and man.positions("flat", camera_id):
        fit = radiometry.noise_from_stacks(_stack(man, "dark", camera_id), _stack(man, "flat", camera_id))
        rep["noise"] = {"read": fit.read, "full_well": fit.full_well}
        rep["noise_fit"] = radiometry.to_dict(fit)
    if man.positions("sweep", camera_id):
        ts, ys = [], []
        for p in man.positions("sweep", camera_id):
            slot = man.slot("sweep", p)
            img = load_linear(slot["all"])[0]
            h, w = img.shape
            ts.append(slot["exposure_s"])
            ys.append(float(np.median(img[h // 4:3 * h // 4, w // 4:3 * w // 4])))
        lin = radiometry.linearity(ts, ys)
        lin["pass"] = lin["r2"] > GATES["linearity_r2"] and lin["max_rel_dev_linear_range"] < GATES["linearity_dev"]
        rep["checks"]["linearity"] = lin
        ok &= lin["pass"]
    if drift_position in man.positions("matte_ball", camera_id):
        slot = man.slot("matte_ball", drift_position)
        amb = load_linear(slot["ambient"])[0] if "ambient" in slot else None
        for combo, pth in slot.get("combos", {}).items():
            if all(i in slot["lights"] for i in combo):
                singles = [load_linear(slot["lights"][i])[0] for i in combo]
                a = radiometry.additivity(load_linear(pth)[0], singles, amb)
                key = "+".join(map(str, combo))
                rep["checks"].setdefault("additivity", {})[key] = {"ratio": a, "pass": abs(a - 1) < GATES["additivity"]}
                ok &= abs(a - 1) < GATES["additivity"]
        dpos = man.positions("drift", camera_id)
        if dpos and drift_light in slot["lights"]:
            end = load_linear(man.slot("drift", dpos[-1])["lights"][drift_light])[0]
            d = radiometry.drift(load_linear(slot["lights"][drift_light])[0], end, amb)
            rep["checks"]["drift"] = {"ratio": d, "pass": abs(d - 1) < GATES["drift"]}
            ok &= rep["checks"]["drift"]["pass"]
    rep["pass"] = bool(ok)
    rep["meta"] = {"calib_manifest": str(calib_manifest), "camera_id": camera_id, "git_hash": git_hash(),
                   "written": datetime.datetime.now().isoformat(timespec="seconds")}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1))
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-manifest", required=True)
    ap.add_argument("--out", required=True, help="radiometry.json")
    ap.add_argument("--camera-id", type=int, default=1)
    a = ap.parse_args()
    r = run(a.calib_manifest, a.out, a.camera_id)
    print(json.dumps({k: r[k] for k in ("noise", "checks", "pass") if k in r}, indent=1, default=str))
