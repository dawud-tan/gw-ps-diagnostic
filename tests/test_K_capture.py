"""K. Capture sessions: the checklist order, the lights on at every capture, manifests the
pipeline reads, the shutter in force at every capture (PS vs all-lights), the gphoto2
driver's commands (fake gphoto2), the serial light protocol, and the example rig config."""
import io
import json
import os
import stat
from pathlib import Path

import pytest

import capture_session
from gwps import capture as cap
from gwps.calib_io import CalibManifest
from gwps.io import Manifest

IDS = [1, 2, 3, 4, 5, 6, 7, 8]
SETTINGS = {"imageformat": "RAW", "iso": "100", "aperture": "11", "shutterspeed": "1/60"}
PILOT = {"camera_settings": SETTINGS, "n_dark": 3, "n_flat": 3, "sweep_shutters": ["1/250", "1/60", "1/15"],
         "n_intrinsics": 2, "ball_positions": ["p0", "p1", "p2"], "additivity_combo": [1, 2],
         "board_placements": ["f+20_h0", "t+30_h0"], "colour_chart": True}


def _session(tmp_path, ext="png"):
    return cap.Session(tmp_path, cap.MockCamera(), cap.MockLights(), cap.MockTurntable(), cap.AutoOperator(),
                       IDS, ext, extra_channels={"backdrop": "backdrop"})


def _shots(s):
    """{relative path: shutter the mock camera was set to when it took that frame}."""
    return {os.path.relpath(p, s.out): st.get("shutterspeed")
            for p, st in zip(s.camera.captures, s.camera.shot_settings)}


def test_pilot_session(tmp_path):
    s = _session(tmp_path)
    cap.run_pilot(s, PILOT)
    man = CalibManifest(tmp_path / "calib_manifest.csv")
    assert man.positions("dark") == ["d0", "d1", "d2"] and len(man.positions("flat")) == 3
    sweep = [man.slot("sweep", p)["exposure_s"] for p in man.positions("sweep")]
    assert sweep == pytest.approx([1 / 250, 1 / 60, 1 / 15])
    assert s.camera.settings["shutterspeed"] == "1/60"                          # restored after the sweep
    for p in ("p0", "p1", "p2"):
        mb = man.slot("matte_ball", p)
        assert sorted(mb["lights"]) == IDS and "silhouette" in mb and "ambient" in mb
        assert sorted(man.slot("mirror_ball", p)["lights"]) == IDS
    assert (1, 2) in man.slot("matte_ball", "p0")["combos"]
    assert man.positions("fabric_board") == ["f+20_h0", "f+20_h0_par", "t+30_h0", "t+30_h0_par"]
    assert sorted(man.slot("colour_chart", "c0")["lights"]) == IDS
    assert man.slot("drift", "end")["lights"][1]
    ev = json.loads((tmp_path / "session.json").read_text())["events"]
    caps = [e for e in ev if e["kind"] == "capture"]
    by = {e["path"]: e["lights"] for e in caps}
    assert by["calib/matte_ball/p0/silhouette.png"] == ["backdrop"]
    assert by["calib/matte_ball/p0/ambient.png"] == []
    assert by["calib/matte_ball/p0/3.png"] == [3]
    assert by["calib/matte_ball/p0/1+2.png"] == [1, 2]
    assert by["calib/dark/d0/all.png"] == []
    prompts = [e["msg"] for e in ev if e["kind"] == "prompt"]
    assert "lens cap on" in prompts[0] and "Drift check" in prompts[-1]
    assert any("MIRROR" in m and "PARALLEL" in m for m in prompts)
    assert len(caps) == len(s.camera.captures) == sum(1 for _ in open(tmp_path / "calib_manifest.csv")) - 1


def test_garment_session_second_pass(tmp_path):
    s = _session(tmp_path)
    cap.run_garment(s, {"camera_settings": SETTINGS, "n_steps": 4, "step_deg": 90.0, "parallel": "second_pass",
                        "metric_board_frames": 2})
    man = Manifest(tmp_path / "manifest.csv")
    par = Manifest(tmp_path / "manifest_parallel.csv")
    assert sorted(man.views) == [f"step{k:04d}.png" for k in range(4)]
    assert all(sorted(v["lights"]) == IDS and v["ambient"] is not None for v in man.views.values())
    assert all(sorted(v["lights"]) == IDS and v["silhouette"] is None for v in par.views.values())
    assert s.turntable.angles == [-40.0, 40.0] + [0.0, 90.0, 180.0, 270.0] * 2   # metric board within +-40
    dev = cap.develop_for_colmap(tmp_path)
    assert [os.path.basename(p) for p in dev] == [f"step{k:04d}.png" for k in range(4)]


def test_garment_silhouettes_and_metric_frames(tmp_path):
    s = _session(tmp_path)
    cap.run_garment(s, {"camera_settings": SETTINGS, "n_steps": 3, "step_deg": 120.0,
                        "metric_board_angles": [-40, 0, 40]})
    man = Manifest(tmp_path / "manifest.csv")
    assert all(v["silhouette"] is not None and v["silhouette"].exists() for v in man.views.values())
    ev = json.loads((tmp_path / "session.json").read_text())["events"]
    by = {e["path"]: e["lights"] for e in ev if e["kind"] == "capture"}
    assert by["steps/step0001/silhouette.png"] == ["backdrop"] and by["steps/step0001/sfm.png"] == IDS
    cm = CalibManifest(tmp_path / "calib_manifest.csv")
    assert cm.positions("metric_board") == ["m0", "m1", "m2"] and cm.slot("metric_board", "m2")["all"].exists()
    assert [e["deg"] for e in ev if e["kind"] == "turntable"] == [-40.0, 0.0, 40.0, 0.0, 120.0, 240.0]
    s2 = _session(tmp_path / "black")                    # black backdrop: masks come from the SfM frames
    cap.run_garment(s2, {"camera_settings": SETTINGS, "n_steps": 2, "step_deg": 180.0, "mask_source": "sfm"})
    assert all(v["silhouette"] is None for v in Manifest(tmp_path / "black/manifest.csv").views.values())
    with pytest.raises(ValueError, match="mask_source"):
        cap.run_garment(_session(tmp_path / "bad"), {"camera_settings": SETTINGS, "n_steps": 1, "step_deg": 90.0,
                                                    "mask_source": "chroma"})


def test_pilot_all_lights_shutter(tmp_path):
    s = _session(tmp_path)
    cap.run_pilot(s, dict(PILOT, all_lights_shutter="1/500"))
    shot = _shots(s)
    bright = {p for p in shot if p.startswith(("calib/flat/", "calib/intrinsics/"))}
    sweep = {p for p in shot if p.startswith("calib/sweep/")}
    assert len(bright) == 3 + 2 and {shot[p] for p in bright} == {"1/500"}
    assert [shot[f"calib/sweep/e{k}/all.png"] for k in range(3)] == ["1/250", "1/60", "1/15"]
    rest = set(shot) - bright - sweep                     # darks, balls, the 1+2 pair, board, chart, drift
    assert {"calib/dark/d0/all.png", "calib/matte_ball/p0/1+2.png", "calib/matte_ball/p0/silhouette.png"} <= rest
    assert {shot[p] for p in rest} == {"1/60"}
    js = json.loads((tmp_path / "session.json").read_text())
    assert js["shutter"] == {"setting": "shutterspeed", "ps": "1/60", "all_lights": "1/500"}
    assert all(e["shutter"] == shot[e["path"]] for e in js["events"] if e["kind"] == "capture")


def test_garment_all_lights_shutter(tmp_path):
    s = _session(tmp_path)
    cap.run_garment(s, {"camera_settings": SETTINGS, "all_lights_shutter": "1/500", "n_steps": 3, "step_deg": 120.0,
                        "parallel": "per_step", "metric_board_frames": 2})
    shot = _shots(s)
    bright = {p for p in shot if p.endswith("/sfm.png") or p.startswith("calib/metric_board/")}
    assert len(bright) == 3 + 2 and {shot[p] for p in bright} == {"1/500"}
    assert {shot[p] for p in set(shot) - bright} == {"1/60"}      # silhouette, ambient, crossed and parallel lights
    ev = json.loads((tmp_path / "session.json").read_text())["events"]
    assert len([e for e in ev if e["kind"] == "shutter"]) == 2 * 3  # the metric frames share step 0's change
    assert Manifest(tmp_path / "manifest.csv").views                # manifests are unchanged


def test_garment_exposure_brackets(tmp_path):
    """A dark garment: ambient and every PS light again at each longer shutter, every step; the
    manifest gains exposure_s and the pipeline reads the brackets back."""
    s = _session(tmp_path)
    cap.run_garment(s, {"camera_settings": SETTINGS, "all_lights_shutter": "1/500", "n_steps": 2, "step_deg": 180.0,
                        "parallel": "per_step", "ps_brackets": ["1/15", "1/4"]})
    shot = _shots(s)
    man = Manifest(tmp_path / "manifest.csv")
    for k in range(2):
        v = man.views[f"step{k:04d}.png"]
        assert v["exposure_s"] == pytest.approx(1 / 60) and sorted(v["lights"]) == IDS
        assert [b["exposure_s"] for b in v["brackets"]] == pytest.approx([1 / 15, 1 / 4])
        assert all(sorted(b["lights"]) == IDS and b["ambient"] is not None for b in v["brackets"])
        st = f"steps/step{k:04d}"
        assert (shot[f"{st}/sfm.png"], shot[f"{st}/3.png"], shot[f"{st}/3_b1.png"], shot[f"{st}/ambient_b2.png"]) == \
            ("1/500", "1/60", "1/15", "1/4")
    assert len(shot) == 2 * (1 + 1 + 1 + 8 + 8 + 2 * 9)            # sfm, silhouette, ambient, crossed, parallel, brackets
    assert Manifest(tmp_path / "manifest_parallel.csv").views["step0000.png"]["brackets"] == []
    ev = json.loads((tmp_path / "session.json").read_text())["events"]
    by = {e["path"]: e["lights"] for e in ev if e["kind"] == "capture"}
    assert by["steps/step0001/ambient_b1.png"] == [] and by["steps/step0001/4_b2.png"] == [4]
    with pytest.raises(ValueError, match="longer than the PS shutter"):
        cap.run_garment(_session(tmp_path / "short"), {"camera_settings": SETTINGS, "n_steps": 1, "step_deg": 90.0,
                                                      "ps_brackets": ["1/125"]})
    with pytest.raises(ValueError, match="need the PS shutter"):
        cap.run_garment(_session(tmp_path / "none"), {"camera_settings": {"iso": "100"}, "n_steps": 1, "step_deg": 90.0,
                                                     "ps_brackets": ["1"]})


def test_two_lights_stay_at_the_ps_shutter(tmp_path):
    s = _session(tmp_path)
    cap.run_garment(s, {"camera_settings": SETTINGS, "all_lights_shutter": "1/500", "n_steps": 1, "step_deg": 90.0,
                        "sfm_lights": [1, 5]})
    assert set(_shots(s).values()) == {"1/60"}
    assert s.shutter_for({1, 2, 3}) == "1/500" and s.shutter_for({"backdrop", 1}) == "1/60"


def test_shutters_need_the_ps_shutter(tmp_path):
    with pytest.raises(ValueError, match="all_lights_shutter"):
        _session(tmp_path / "a").lock({"iso": "100"}, all_lights_shutter="1/500")
    with pytest.raises(ValueError, match="sweep"):
        cap.run_pilot(_session(tmp_path / "b"), dict(PILOT, camera_settings={"iso": "100"}))
    s = _session(tmp_path / "c")                         # no shutters at all: the camera is left alone
    cap.run_garment(s, {"camera_settings": {"iso": "100"}, "n_steps": 1, "step_deg": 90.0})
    assert set(_shots(s).values()) == {None}


FAKE_GPHOTO2 = r"""#!/usr/bin/env bash
state="$FAKE_GP_STATE"; echo "$@" >> "$state.log"
case "$1" in
  --set-config) k="${2%%=*}"; v="${2#*=}"; [ "$k" = "iso" ] && [ "$v" = "50" ] && v="100"
                [ "$k" = "shutterspeed" ] && [ "$v" = "1/9000" ] && v="1/8000"; echo "$k=$v" >> "$state" ;;
  --get-config) v=$(grep "^$2=" "$state" | tail -1 | cut -d= -f2-); echo "Label: $2"; echo "Current: $v" ;;
  --capture-image-and-download) printf 'RAW' > "$3" ;;
esac
"""


def _fake_gphoto2(tmp_path, monkeypatch):
    exe = tmp_path / "gphoto2"
    exe.write_text(FAKE_GPHOTO2)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("FAKE_GP_STATE", str(tmp_path / "state"))
    return exe


def test_gphoto2_driver(tmp_path, monkeypatch):
    exe = _fake_gphoto2(tmp_path, monkeypatch)
    cam = cap.Gphoto2Camera(str(exe))
    assert cam.configure(SETTINGS) == SETTINGS
    p = cam.capture(tmp_path / "shots/a.cr3")
    assert p.read_text() == "RAW"
    log = (tmp_path / "state.log").read_text()
    assert "--capture-image-and-download --filename" in log and "--set-config iso=100" in log
    with pytest.raises(RuntimeError, match="camera reports"):
        cam.configure({"iso": "50"})                       # the fake camera silently keeps 100


def test_gphoto2_shutter_changes(tmp_path, monkeypatch):
    exe = _fake_gphoto2(tmp_path, monkeypatch)

    def session(name):
        return cap.Session(tmp_path / name, cap.Gphoto2Camera(str(exe)), cap.MockLights(), cap.MockTurntable(),
                           cap.AutoOperator(), IDS, "cr3")

    s = session("a")
    cap.run_garment(s, {"camera_settings": SETTINGS, "all_lights_shutter": "1/500", "n_steps": 2, "step_deg": 180.0})
    log = (tmp_path / "state.log").read_text().splitlines()
    cur, at = None, {}                                   # replay the camera's commands: shutter at each capture
    for ln in log:
        if ln.startswith("--set-config shutterspeed="):
            cur = ln.split("=", 1)[1]
        elif ln.startswith("--capture-image-and-download"):
            at[os.path.relpath(ln.split()[2], s.out)] = cur
    assert at["steps/step0000/sfm.cr3"] == at["steps/step0001/sfm.cr3"] == "1/500"
    assert {v for k, v in at.items() if not k.endswith("sfm.cr3")} == {"1/60"} and len(at) == 2 * 11   # + silhouette
    assert log.count("--get-config shutterspeed") == 1 + 4  # the lock, then every change is read back
    with pytest.raises(RuntimeError, match="camera reports"):  # the fake camera snaps 1/9000 to 1/8000
        cap.run_garment(session("b"), {"camera_settings": SETTINGS, "all_lights_shutter": "1/9000",
                                       "n_steps": 1, "step_deg": 90.0})


def test_rig_example_config(tmp_path):
    cfg = json.loads((Path(__file__).resolve().parents[1] / "config/rig.example.json").read_text())
    pilot, garment = capture_session.plan_for(cfg, "pilot"), capture_session.plan_for(cfg, "garment")
    ps = pilot["camera_settings"][pilot["shutter_setting"]]
    assert 6 <= cap.shutter_seconds(ps) / cap.shutter_seconds(pilot["all_lights_shutter"]) <= 10   # ~3 stops
    t = [cap.shutter_seconds(x) for x in pilot["sweep_shutters"]]
    assert min(t) < cap.shutter_seconds(pilot["all_lights_shutter"]) < max(t)
    assert capture_session.plan_for(dict(cfg, garment=dict(cfg["garment"], all_lights_shutter="1/30")),
                                     "garment")["all_lights_shutter"] == "1/30"                    # plan wins
    for name, plan in (("pilot", pilot), ("garment", garment)):     # the whole plan runs (mock drivers)
        s = cap.Session(tmp_path / name, cap.MockCamera(), cap.MockLights(), cap.MockTurntable(),
                        cap.AutoOperator(), cfg["ps_light_ids"], "png", extra_channels=cfg["extra_channels"])
        (cap.run_pilot if name == "pilot" else cap.run_garment)(s, plan)
        assert {"0.5", "1/15"} <= set(_shots(s).values())


def test_serial_lights_protocol():
    buf = io.BytesIO()
    lights = cap.SerialLights(stream=buf, settle_s=0, verify=False)             # the bytes only (replies: test_L_*)
    lights.set({3})
    lights.set({"backdrop"})
    assert buf.getvalue().decode().splitlines() == ["ALL=0", "L3=1", "ALL=0", "Lbackdrop=1"]
