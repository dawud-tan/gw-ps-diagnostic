"""Run a capture session (pilot or garment) following docs/capture_checklist.md; writes the
manifests (calib_manifest.csv / manifest.csv / manifest_parallel.csv) and session.json.

  capture_session.py --config config/rig.json --plan pilot --out captures/pilot_2026-10-01
  capture_session.py --config config/rig.json --plan garment --out captures/shirt01
  --mock runs everything with simulated camera, lights and turntable (a dry run of the order).
After a garment session, `--develop` writes colmap_images/stepNNNN.png from the SfM RAWs; then
stage_masks.py, stage_sfm.py and stage1_gw_prep.py --masks (docs/capture_checklist.md).

The camera section's `settings` hold the PS shutter under `shutter_setting` (default
"shutterspeed"); `all_lights_shutter` is the shorter one for frames lit by more than two PS
lights (SfM, metric board, flats, intrinsics). A plan may override either key.
"""
import argparse
import json
from pathlib import Path

from gwps import capture as cap


def build(cfg, mock, operator):
    if mock:
        return cap.MockCamera(), cap.MockLights(), cap.MockTurntable()
    c = cfg["camera"]
    camera = cap.Gphoto2Camera(c.get("exe", "gphoto2"), c.get("port"))
    L = cfg["lights"]
    if L["driver"] == "serial":
        lights = cap.SerialLights(L["port"], L.get("baud", 115200), L.get("on", "L{id}=1"), L.get("off", "L{id}=0"),
                                  L.get("all_off", "ALL=0"), L.get("settle_s", 0.3), verify=L.get("verify", True))
    else:
        lights = cap.ManualLights(operator)
    T = cfg.get("turntable", {"driver": "manual"})
    if T["driver"] == "serial":
        tt = cap.SerialTurntable(T["port"], T.get("baud", 115200), T.get("command", "ROT {deg:.3f}"), T.get("settle_s", 2.0))
    else:
        tt = cap.ManualTurntable(operator)
    return camera, lights, tt


def plan_for(cfg, name):
    """The named plan with the camera's settings and shutters; keys in the plan win."""
    cam = cfg["camera"]
    return {"shutter_setting": cam.get("shutter_setting", "shutterspeed"),
            "all_lights_shutter": cam.get("all_lights_shutter"),
            **cfg[name], "camera_settings": cam["settings"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--plan", choices=["pilot", "garment"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--yes", action="store_true", help="auto-confirm prompts (dry runs only)")
    ap.add_argument("--develop", action="store_true", help="develop SfM RAWs of a garment session for COLMAP")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    if a.develop:
        print("\n".join(cap.develop_for_colmap(a.out)))
        return
    operator = cap.AutoOperator() if (a.yes or a.mock) else cap.ConsoleOperator()
    camera, lights, tt = build(cfg, a.mock, operator)
    s = cap.Session(a.out, camera, lights, tt, operator, cfg["ps_light_ids"], cfg["camera"].get("ext", "cr3"),
                    cfg.get("camera_id", 1), cfg.get("extra_channels"))
    (cap.run_pilot if a.plan == "pilot" else cap.run_garment)(s, plan_for(cfg, a.plan))
    print(f"session written to {a.out}")


if __name__ == "__main__":
    main()
