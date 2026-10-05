"""Capture sessions that follow docs/capture_checklist.md and write the manifests as they go.

The file naming is defined here (it is no longer a ⚠ item):
  pilot:   <session>/calib/<target>/<position>/<light_id>.<ext>  -> calib_manifest.csv
  garment: <session>/steps/stepNNNN/<light_id>.<ext>             -> manifest.csv
           <session>/steps/stepNNNN/sfm.<ext>  (the SfM/GW image; developed to
           <session>/colmap_images/stepNNNN.png by develop_for_colmap)
           <session>/steps/stepNNNN/silhouette.<ext>  (backlit frame for the object mask;
           in manifest.csv as light_id 'silhouette')
           <session>/steps/stepNNNN/<light_id>_b<j>.<ext>, ambient_b<j>.<ext>  (exposure bracket j
           of a dark garment; manifest.csv then has an exposure_s column on every row)
           parallel-polariser frames                              -> manifest_parallel.csv
           <session>/calib/metric_board/m<k>/all.<ext>  (ChArUco board on the turntable)
                                                                 -> calib_manifest.csv
light_id: a PS light id, 'ambient', 'silhouette', 'all', or a combination such as '1+2'.

Drivers are small and pluggable: a camera (gphoto2 CLI, or mock), lights (serial text
protocol, manual prompts, or mock), a turntable (serial, manual, or mock) and an operator
who confirms manual steps (console, or auto for tests/dry runs). Every event is logged with a
timestamp to session.json, with the camera settings read back after locking them.

Shutters: the PS shutter is the one locked with the camera settings, set so single-light
frames stay below ~90 % of full scale. A frame lit by more than two PS lights at once (the SfM
and metric-board frames, the flats, the intrinsics shots) is 2.5-2.8 stops brighter with all
eight on and would clip, so it uses the plan's `all_lights_shutter` when one is set. The
additivity pair stays at the PS shutter on purpose, and the exposure sweep sets its own.
Every shutter change is read back from the camera, and every capture logs its shutter.

Dark garments: a garment plan's `ps_brackets` (shutters longer than the PS shutter) reshoots
the ambient frame and every PS light at each of them, every step; stage 4 merges each light per
pixel to the longest exposure that does not clip. Keep the PS shutter itself at the pilot's: E
and the colour correction were measured there, and albedo comes out in its units. Each bracket
adds 9 frames per step at a longer shutter, so a session grows by roughly that share.
"""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from pathlib import Path


# ---------------------------------------------------------------- operator
class ConsoleOperator:
    def prompt(self, msg):
        input(f"\n>>> {msg}\n    [Enter when done] ")


class AutoOperator:
    """Confirms every prompt (tests, dry runs); keeps the prompts for inspection."""

    def __init__(self):
        self.prompts = []

    def prompt(self, msg):
        self.prompts.append(msg)


# ---------------------------------------------------------------- cameras
class Gphoto2Camera:
    """Tethered camera through the gphoto2 command-line tool (libgphoto2). Setting names are
    brand-specific (Canon: iso, aperture, shutterspeed, imageformat, whitebalance, capturetarget;
    see `gphoto2 --list-config`), so they come from the rig config."""

    def __init__(self, exe="gphoto2", port=None, timeout=60):
        self.exe, self.port, self.timeout = exe, port, timeout

    def _run(self, *args):
        cmd = [self.exe] + (["--port", self.port] if self.port else []) + list(args)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout)
        if r.returncode != 0:
            raise RuntimeError(f"{' '.join(cmd)} failed: {r.stderr.strip()[-500:]}")
        return r.stdout

    def set(self, name, value):
        self._run("--set-config", f"{name}={value}")

    def get(self, name):
        for ln in self._run("--get-config", name).splitlines():
            if ln.startswith("Current:"):
                return ln.split(":", 1)[1].strip()
        return None

    def configure(self, settings):
        """Lock the given settings and read them back (a mismatch is an error)."""
        back = {}
        for k, v in settings.items():
            self.set(k, v)
            back[k] = self.get(k)
            if back[k] is not None and str(back[k]) != str(v):
                raise RuntimeError(f"camera setting {k}: asked {v!r}, camera reports {back[k]!r}")
        return back

    def capture(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._run("--capture-image-and-download", "--filename", str(path), "--force-overwrite")
        if not path.exists():
            raise RuntimeError(f"capture did not produce {path}")
        return path


class MockCamera:
    """Writes a tiny 16-bit PNG per capture; records settings, captures and the settings in
    force at each capture."""

    def __init__(self):
        self.settings, self.captures, self.shot_settings = {}, [], []

    def set(self, name, value):
        self.settings[name] = str(value)

    def get(self, name):
        return self.settings.get(name)

    def configure(self, settings):
        for k, v in settings.items():
            self.set(k, v)
        return dict(self.settings)

    def capture(self, path):
        import numpy as np
        import cv2
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path.with_suffix(".png")) if path.suffix.lower() != ".png" else str(path),
                    np.full((8, 8), 1000 + len(self.captures), np.uint16))
        if path.suffix.lower() != ".png":
            shutil.move(str(path.with_suffix(".png")), path)
        self.captures.append(str(path))
        self.shot_settings.append(dict(self.settings))
        return path


# ---------------------------------------------------------------- lights
class SerialLights:
    """Relay/LED controller speaking a line protocol over a serial port (pyserial), e.g. the Pico 2
    firmware in firmware/pico2_lights: on="L{id}=1", off="L{id}=0", all_off="ALL=0". Channel names
    beyond the PS ids (e.g. 'backdrop') are allowed. `stream` replaces the port in tests.

    verify (default): every command must be answered "OK", and after switching, the state query
    ("?" -> "STATE 3,backdrop") must list exactly the lights asked for, as camera settings are
    read back; a controller that missed a command would otherwise light a frame wrongly and
    silently. A controller without replies needs verify=False."""

    def __init__(self, port=None, baud=115200, on="L{id}=1", off="L{id}=0", all_off="ALL=0",
                 settle_s=0.3, stream=None, verify=True, state_query="?"):
        if stream is None:
            import serial
            stream = serial.Serial(port, baud, timeout=2)
        self.stream, self.fmt_on, self.fmt_off, self.fmt_all_off = stream, on, off, all_off
        self.settle_s, self.state = settle_s, set()
        self.verify, self.state_query = verify, state_query

    def _line(self, line):
        self.stream.write((line + "\n").encode())
        if hasattr(self.stream, "flush"):
            self.stream.flush()
        if not self.verify:
            return None
        return self.stream.readline().decode(errors="replace").strip()

    def _send(self, line):
        reply = self._line(line)
        if self.verify and not (reply or "").startswith("OK"):
            raise RuntimeError(f"light controller: {line!r} -> {reply or 'no reply'!r}")

    def set(self, on):
        on = set(on)
        self._send(self.fmt_all_off)
        for i in sorted(on, key=str):
            self._send(self.fmt_on.format(id=i))
        if self.verify and self.state_query:
            reply = self._line(self.state_query)
            got = reply.split()[1] if reply.startswith("STATE ") and len(reply.split()) > 1 else None
            got = set() if got == "-" else (set(got.split(",")) if got else None)
            if got != {str(i) for i in on}:
                raise RuntimeError(f"light controller reports {reply!r}, asked for {sorted(on, key=str)}")
        self.state = on
        time.sleep(self.settle_s)


class ManualLights:
    def __init__(self, operator):
        self.operator, self.state = operator, set()

    def set(self, on):
        on = set(on)
        if on != self.state:
            self.operator.prompt(f"Lights: switch ON only {sorted(on, key=str) or 'nothing (all off)'}")
        self.state = on


class MockLights:
    def __init__(self):
        self.state, self.history = set(), []

    def set(self, on):
        self.state = set(on)
        self.history.append(sorted(self.state, key=str))


# ---------------------------------------------------------------- turntables
class SerialTurntable:
    def __init__(self, port=None, baud=115200, command="ROT {deg:.3f}", settle_s=2.0, stream=None):
        if stream is None:
            import serial
            stream = serial.Serial(port, baud, timeout=10)
        self.stream, self.command, self.settle_s = stream, command, settle_s

    def rotate_to(self, deg):
        self.stream.write((self.command.format(deg=deg) + "\n").encode())
        time.sleep(self.settle_s)


class ManualTurntable:
    def __init__(self, operator):
        self.operator = operator

    def rotate_to(self, deg):
        self.operator.prompt(f"Turntable: rotate to {deg:.1f} deg")


class MockTurntable:
    def __init__(self):
        self.angles = []

    def rotate_to(self, deg):
        self.angles.append(float(deg))


# ---------------------------------------------------------------- session
def shutter_seconds(s):
    s = str(s).strip().rstrip("s")
    if "/" in s:
        a, b = s.split("/")
        return float(a) / float(b)
    return float(s)


class Session:
    def __init__(self, out, camera, lights, turntable, operator, ps_ids, ext="cr3", camera_id=1,
                 extra_channels=None):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.camera, self.lights, self.turntable, self.operator = camera, lights, turntable, operator
        self.ps_ids, self.ext, self.camera_id = list(ps_ids), ext.lstrip("."), camera_id
        self.channels = extra_channels or {}
        self.events, self._files = [], {}
        self.settings = {}
        self.shutter_setting, self.ps_shutter, self.all_lights_shutter = "shutterspeed", None, None
        self._shutter = None                               # the shutter the camera is set to

    # -- bookkeeping
    def log(self, kind, **kw):
        self.events.append({"t": time.time(), "kind": kind, **kw})

    def _writer(self, name, header):
        if name not in self._files:
            f = open(self.out / name, "w", newline="")
            w = csv.writer(f)
            w.writerow(header)
            self._files[name] = (f, w)
        return self._files[name][1]

    def _row(self, name, header, row):
        self._writer(name, header).writerow(row)
        self._files[name][0].flush()

    def ask(self, msg):
        self.log("prompt", msg=msg)
        self.operator.prompt(msg)

    def rotate(self, deg):
        self.turntable.rotate_to(deg)
        self.log("turntable", deg=float(deg))

    def lock(self, settings, shutter_setting="shutterspeed", all_lights_shutter=None):
        """Lock the camera settings (read back). Their `shutter_setting` entry is the PS shutter."""
        ps = settings.get(shutter_setting)
        if all_lights_shutter and ps is None:
            raise ValueError(f"all_lights_shutter needs the PS shutter ({shutter_setting!r}) in the camera "
                             "settings, to return to it after each all-lights frame")
        self.settings = self.camera.configure(settings)
        self.shutter_setting = shutter_setting
        self.ps_shutter = None if ps is None else str(ps)
        self.all_lights_shutter = str(all_lights_shutter) if all_lights_shutter else None
        self._shutter = self.ps_shutter
        self.log("camera_settings", settings=self.settings, shutter_setting=shutter_setting,
                 ps_shutter=self.ps_shutter, all_lights_shutter=self.all_lights_shutter)

    def shutter_for(self, on):
        """The all-lights shutter (if set) for a frame lit by more than two PS lights, else the
        PS shutter. None when no PS shutter was locked: the camera is left as it is."""
        if self.all_lights_shutter and len(set(on) & set(self.ps_ids)) > 2:
            return self.all_lights_shutter
        return self.ps_shutter

    def _use_shutter(self, value):
        if value is None or str(value) == self._shutter:
            return
        value = str(value)
        self.camera.set(self.shutter_setting, value)
        back = self.camera.get(self.shutter_setting)
        if back is not None and str(back) != value:
            raise RuntimeError(f"camera setting {self.shutter_setting}: asked {value!r}, camera reports {back!r}")
        self._shutter = value
        self.log("shutter", value=value)

    def shoot(self, rel, on, shutter=None):
        """One frame with exactly the lights in `on`, at `shutter` if given (the exposure
        sweep), else at shutter_for(on)."""
        self._use_shutter(shutter if shutter is not None else self.shutter_for(on))
        self.lights.set(on)
        path = self.camera.capture(self.out / rel)
        self.log("capture", path=str(rel), lights=sorted(on, key=str), shutter=self._shutter)
        return path

    def close(self):
        self.lights.set(set())
        for f, _ in self._files.values():
            f.close()
        (self.out / "session.json").write_text(json.dumps(
            {"camera_settings": self.settings,
             "shutter": {"setting": self.shutter_setting, "ps": self.ps_shutter, "all_lights": self.all_lights_shutter},
             "ps_ids": self.ps_ids, "events": self.events}, indent=1))

    # -- calibration rows
    CALIB_HEADER = ["target", "position", "camera_id", "light_id", "path", "exposure_s"]

    def calib(self, target, position, light_id, on, exposure_s="", shutter=None):
        rel = f"calib/{target}/{position}/{light_id}.{self.ext}"
        self.shoot(rel, on, shutter)
        self._row("calib_manifest.csv", self.CALIB_HEADER,
                  [target, position, self.camera_id, light_id, rel, exposure_s])

    def calib_lights(self, target, position, ambient=True):
        if ambient:
            self.calib(target, position, "ambient", set())
        for i in self.ps_ids:
            self.calib(target, position, i, {i})


def run_pilot(s: Session, plan):
    """The pilot session of docs/capture_checklist.md, in its order."""
    shutter_setting = plan.get("shutter_setting", "shutterspeed")
    if plan.get("sweep_shutters") and plan["camera_settings"].get(shutter_setting) is None:
        raise ValueError(f"the exposure sweep needs the PS shutter ({shutter_setting!r}) in the camera "
                         "settings, to return to it after the sweep")
    s.lock(plan["camera_settings"], shutter_setting, plan.get("all_lights_shutter"))
    backdrop = s.channels.get("backdrop", "backdrop")
    flat_on = set(plan.get("flat_lights", s.ps_ids))
    # 1. noise and linearity (flats lit by more than two lights use the all-lights shutter)
    s.ask("Noise frames: put the lens cap on.")
    for k in range(plan.get("n_dark", 10)):
        s.calib("dark", f"d{k}", "all", set())
    s.ask("Remove the lens cap. Place the evenly lit matte white card filling the frame.")
    for k in range(plan.get("n_flat", 10)):
        s.calib("flat", f"f{k}", "all", flat_on)
    for k, sh in enumerate(plan.get("sweep_shutters", [])):
        s.calib("sweep", f"e{k}", "all", flat_on, exposure_s=repr(shutter_seconds(sh)), shutter=sh)
    # 2. intrinsics
    for k in range(plan.get("n_intrinsics", 15)):
        s.ask(f"Intrinsics shot {k + 1}/{plan.get('n_intrinsics', 15)}: move the ChArUco board "
              "(0.7-1.1x garment distance, tilt up to 35 deg, reach a new edge or corner of the frame).")
        s.calib("intrinsics", f"i{k}", "all", set(s.ps_ids))
    # 3. balls
    combo = plan.get("additivity_combo", s.ps_ids[:2])
    for k, pos in enumerate(plan.get("ball_positions", [f"p{i}" for i in range(6)])):
        s.ask(f"Ball position {pos}: seat the MATTE ball. Lens polariser CROSSED.")
        s.calib("matte_ball", pos, "silhouette", {backdrop})
        s.calib_lights("matte_ball", pos)
        if k == 0 and combo:
            s.calib("matte_ball", pos, "+".join(map(str, combo)), set(combo))
        s.ask(f"Ball position {pos}: swap in the MIRROR ball without touching the seat. Lens polariser PARALLEL.")
        s.calib_lights("mirror_ball", pos)
        s.ask("Lens polariser back to CROSSED.")
    # 4. fabric board, crossed then parallel
    for pl in plan.get("board_placements", []):
        s.ask(f"Fabric board placement {pl}. Lens polariser CROSSED.")
        s.calib_lights("fabric_board", pl)
        s.ask(f"Placement {pl}: lens polariser PARALLEL.")
        s.calib_lights("fabric_board", f"{pl}_par")
        s.ask("Lens polariser back to CROSSED.")
    # 5. colour
    if plan.get("colour_chart", True):
        s.ask("ColorChecker and the reflectance reference at the garment position. Lens polariser CROSSED.")
        s.calib_lights("colour_chart", "c0")
    # 6. drift
    first = plan.get("ball_positions", ["p0"])[0]
    s.ask(f"Drift check: seat the MATTE ball at {first} again.")
    s.calib("drift", "end", s.ps_ids[0], {s.ps_ids[0]})
    s.close()


def metric_board_angles(plan):
    """Turntable angles for the metric-board frames, relative to the board facing the camera.
    A flat board is unreadable edge-on or from behind, so the frames stay within +-40 deg:
    `metric_board_angles` if given, else `metric_board_frames` angles spread over -40..40."""
    if plan.get("metric_board_angles") is not None:
        return [float(a) for a in plan["metric_board_angles"]]
    n = int(plan.get("metric_board_frames", 0))
    return [0.0] if n == 1 else [-40.0 + 80.0 * k / (n - 1) for k in range(n)]


def run_garment(s: Session, plan):
    """Per turntable step: SfM image, the backlit silhouette for the object mask (unless the
    plan's mask_source is 'sfm' (black backdrop, masks from the SfM image) or 'none'), ambient,
    each PS light (crossed); optionally the same lights with the polariser parallel, per step or
    as a second full revolution. The SfM and metric-board frames use the all-lights shutter when
    the plan sets one (two shutter changes per step); the silhouette (backdrop only) uses the PS
    shutter, where the backdrop may saturate.

    Metric board (for stage_sfm.py): before the garment goes on, the ChArUco board stands upright
    on the turntable facing the camera and is shot at metric_board_angles(plan); the frames go
    to the session's calib_manifest.csv as target metric_board, positions m0, m1, ..."""
    s.lock(plan["camera_settings"], plan.get("shutter_setting", "shutterspeed"), plan.get("all_lights_shutter"))
    n, step = plan["n_steps"], plan["step_deg"]
    brackets = [str(b) for b in plan.get("ps_brackets") or []]
    if brackets:
        if s.ps_shutter is None:
            raise ValueError(f"ps_brackets need the PS shutter ({s.shutter_setting!r}) in the camera settings")
        t0 = shutter_seconds(s.ps_shutter)
        short = [b for b in brackets if not shutter_seconds(b) > t0]
        if short or len(set(map(shutter_seconds, brackets))) != len(brackets):
            raise ValueError(f"ps_brackets {brackets}: each must be a distinct shutter longer than the PS "
                             f"shutter {s.ps_shutter} (lengthen a dark garment's exposure, never shorten it)")
    sfm_on = set(plan.get("sfm_lights", s.ps_ids))
    parallel = plan.get("parallel", "none")            # none | per_step | second_pass
    mask_source = plan.get("mask_source", "silhouette")
    if mask_source not in ("silhouette", "sfm", "none"):
        raise ValueError(f"mask_source {mask_source!r}: expected 'silhouette', 'sfm' or 'none'")
    backdrop = s.channels.get("backdrop", "backdrop")
    head = ["step", "camera_id", "colmap_image_name", "light_id", "path"]
    if brackets:                                       # every row then says its exposure
        head = head + ["exposure_s"]
        e0 = [repr(shutter_seconds(s.ps_shutter))]
    else:
        e0 = []
    s.log("plan", n_steps=n, step_deg=step, mask_source=mask_source, parallel=parallel, ps_brackets=brackets)
    angles = metric_board_angles(plan)
    if angles:
        s.ask("Metric check: stand the ChArUco board upright on the turntable next to the mannequin, facing "
              "the camera. Don't touch it until the last metric frame.")
        for k, a in enumerate(angles):
            s.rotate(a)
            s.calib("metric_board", f"m{k}", "all", sfm_on)
        s.ask("Remove the ChArUco board from the turntable.")
    s.ask("Garment on the mannequin, centred on the turntable. Lens polariser CROSSED.")

    def lights_at(k, parallel_pass):
        name = f"step{k:04d}.png"
        man = "manifest_parallel.csv" if parallel_pass else "manifest.csv"
        if not parallel_pass:
            s.shoot(f"steps/step{k:04d}/sfm.{s.ext}", sfm_on)
            if mask_source == "silhouette":
                rel = f"steps/step{k:04d}/silhouette.{s.ext}"
                s.shoot(rel, {backdrop})
                s._row(man, head, [k, s.camera_id, name, "silhouette", rel] + e0)
            rel = f"steps/step{k:04d}/ambient.{s.ext}"
            s.shoot(rel, set())
            s._row(man, head, [k, s.camera_id, name, "ambient", rel] + e0)
        for i in s.ps_ids:
            rel = f"steps/step{k:04d}/{'par_' if parallel_pass else ''}{i}.{s.ext}"
            s.shoot(rel, {i})
            s._row(man, head if not parallel_pass else head[:5], [k, s.camera_id, name, i, rel]
                   + (e0 if not parallel_pass else []))
        if parallel_pass:                              # brackets are for the PS solve: crossed only
            return
        for j, sh in enumerate(brackets, 1):
            e = [repr(shutter_seconds(sh))]
            rel = f"steps/step{k:04d}/ambient_b{j}.{s.ext}"
            s.shoot(rel, set(), shutter=sh)
            s._row(man, head, [k, s.camera_id, name, "ambient", rel] + e)
            for i in s.ps_ids:
                rel = f"steps/step{k:04d}/{i}_b{j}.{s.ext}"
                s.shoot(rel, {i}, shutter=sh)
                s._row(man, head, [k, s.camera_id, name, i, rel] + e)

    for k in range(n):
        s.rotate(k * step)
        lights_at(k, False)
        if parallel == "per_step":
            s.ask("Lens polariser PARALLEL.")
            lights_at(k, True)
            s.ask("Lens polariser back to CROSSED.")
    if parallel == "second_pass":
        s.ask("Second pass: lens polariser PARALLEL (the turntable returns to each step).")
        for k in range(n):
            s.rotate(k * step)
            lights_at(k, True)
        s.ask("Lens polariser back to CROSSED.")
    s.close()


def develop_for_colmap(session_dir, ext="png"):
    """Develop every steps/stepNNNN/sfm.<raw> to colmap_images/stepNNNN.<ext> the same way
    (camera white balance, sRGB, no auto-brightness), so COLMAP/GW see consistent images."""
    import cv2
    session_dir = Path(session_dir)
    out = session_dir / "colmap_images"
    out.mkdir(exist_ok=True)
    done = []
    for d in sorted((session_dir / "steps").glob("step*")):
        src = next(d.glob("sfm.*"), None)
        if src is None:
            continue
        dst = out / f"{d.name}.{ext}"
        if src.suffix.lower() in (".png", ".tif", ".tiff", ".jpg", ".jpeg"):
            shutil.copy(src, dst) if src.suffix.lower() == f".{ext}" else cv2.imwrite(str(dst), cv2.imread(str(src), cv2.IMREAD_UNCHANGED))
        else:
            import rawpy
            with rawpy.imread(str(src)) as raw:
                rgb = raw.postprocess(use_camera_wb=True, no_auto_bright=True, output_bps=8, user_flip=0)
            cv2.imwrite(str(dst), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        done.append(str(dst))
    return done
