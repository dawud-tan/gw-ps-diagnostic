"""L. The Pico 2 light-controller firmware (firmware/pico2_lights/main.py, the file that is flashed) on
the PC: its protocol, the gate levels (low = light on), the auto-off, main()'s wiring against fake
machine/serial modules; and gwps.capture.SerialLights verifying a simulated controller through a
whole capture session, and failing on a controller that ignores a command, says ERR or stays silent."""
import importlib.util
import select
import sys
import time
from pathlib import Path

import pytest

from gwps import capture as cap

ROOT = Path(__file__).resolve().parents[1]


def _firmware():
    spec = importlib.util.spec_from_file_location("pico2_lights", ROOT / "firmware/pico2_lights/main.py")
    fw = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fw)
    return fw


class Clock:
    def __init__(self):
        self.t = 0

    def __call__(self):
        return self.t


def _controller(fw, clock=None):
    levels = {}
    outs = {ch: (lambda v, ch=ch: levels.__setitem__(ch, v)) for ch in fw.PINS}
    return fw.Controller(outs, clock or Clock()), levels


def test_protocol_and_gate_levels():
    fw = _firmware()
    ctl, lv = _controller(fw)
    assert set(lv) == set(fw.PINS) and set(lv.values()) == {1}              # every gate high (off) from the start
    assert ctl.handle("ID?") == "GWPS-LIGHTS 1 channels=1,2,3,4,5,6,7,8,backdrop"
    assert ctl.handle("L3=1") == "OK" and lv["3"] == 0 and lv["4"] == 1     # gate low = light on
    assert ctl.handle("Lbackdrop=1") == "OK" and ctl.handle("?") == "STATE 3,backdrop"
    assert ctl.handle("L3=0") == "OK" and ctl.handle("?") == "STATE backdrop"
    assert ctl.handle("ALL=0") == "OK" and ctl.handle("?") == "STATE -" and set(lv.values()) == {1}
    assert ctl.handle("") is None and ctl.handle("  L1=1 \r") == "OK"
    for bad in ("L9=1", "L1=2", "ALL=1", "X", "AUTO=x", "AUTO=-1", "L=1"):
        assert ctl.handle(bad).startswith("ERR"), bad


def test_auto_off():
    fw = _firmware()
    clock = Clock()
    ctl, lv = _controller(fw, clock)
    ctl.handle("L1=1")
    clock.t = 119_000
    ctl.tick()
    assert ctl.handle("?") == "STATE 1"
    clock.t = 121_000
    ctl.tick()
    assert ctl.handle("?") == "STATE - auto-off" and lv["1"] == 1
    assert ctl.handle("ALL=0") == "OK" and ctl.handle("?") == "STATE -"
    assert ctl.handle("AUTO=0") == "OK"
    ctl.handle("L2=1")
    clock.t = 10_000_000
    ctl.tick()
    assert ctl.handle("?") == "STATE 2"                                      # AUTO=0: never


def test_main_wiring(monkeypatch, capsys):
    fw = _firmware()
    pins = {}

    class Pin:
        OUT = 1

        def __init__(self, id, mode, value=None):
            self.id, self.v = id, value
            pins[id] = self

        def value(self, v=None):
            if v is None:
                return self.v
            self.v = v

    class Poll:
        def register(self, *a):
            pass

        def poll(self, ms):
            return True

    class Stdin:
        text = iter("ID?\nL3=1\nLbackdrop=1\n?\n")

        def read(self, n):
            try:
                return next(self.text)
            except StopIteration:
                raise KeyboardInterrupt                                         # as Ctrl-C would on the Pico

    monkeypatch.setitem(sys.modules, "machine", type(sys)("machine"))
    sys.modules["machine"].Pin = Pin
    monkeypatch.setattr(select, "poll", Poll)
    monkeypatch.setattr(sys, "stdin", Stdin())
    monkeypatch.setattr(time, "ticks_ms", lambda: 0, raising=False)
    monkeypatch.setattr(time, "ticks_diff", lambda a, b: a - b, raising=False)
    with pytest.raises(KeyboardInterrupt):
        fw.main()
    out = capsys.readouterr().out.splitlines()
    assert out == ["GWPS-LIGHTS 1 channels=1,2,3,4,5,6,7,8,backdrop", "OK", "OK", "STATE 3,backdrop"]
    assert pins[4].v == 0 and pins[10].v == 0 and pins[2].v == 1 and pins["LED"].v == 1   # GP4 = light 3, GP10 = backdrop


class FakeSerial:
    """A controller on the other end of the USB serial line (replies end in \\r\\n, as MicroPython's do)."""

    def __init__(self, ctl, silent=False):
        self.ctl, self.silent, self.buf, self.replies = ctl, silent, b"", []

    def write(self, data):
        self.buf += data
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            r = self.ctl.handle(line.decode())
            if r is not None and not self.silent:
                self.replies.append((r + "\r\n").encode())

    def flush(self):
        pass

    def readline(self):
        return self.replies.pop(0) if self.replies else b""


def test_session_through_a_simulated_controller(tmp_path):
    fw = _firmware()
    ctl, _ = _controller(fw)
    lights = cap.SerialLights(stream=FakeSerial(ctl), settle_s=0)

    class Camera(cap.MockCamera):
        def capture(self, path):
            self.lit = getattr(self, "lit", [])
            self.lit.append(set(ctl.on))
            return super().capture(path)

    s = cap.Session(tmp_path, Camera(), lights, cap.MockTurntable(), cap.AutoOperator(), list(range(1, 9)), "png",
                    extra_channels={"backdrop": "backdrop"})
    cap.run_garment(s, {"camera_settings": {"shutterspeed": "0.5"}, "n_steps": 2, "step_deg": 180.0,
                        "metric_board_angles": [0]})
    import json
    ev = [e for e in json.loads((tmp_path / "session.json").read_text())["events"] if e["kind"] == "capture"]
    assert len(ev) == len(s.camera.lit) == 1 + 2 * 11
    for e, lit in zip(ev, s.camera.lit):
        assert lit == {str(x) for x in e["lights"]}, e["path"]                  # what was on is what was asked for


def test_serial_lights_catches_a_bad_controller():
    fw = _firmware()

    class Deaf(fw.Controller):
        def switch(self, ch, on):                                              # says OK, switches nothing
            pass

    ctl, _ = _controller(fw)
    deaf = Deaf({ch: (lambda v: None) for ch in fw.PINS}, Clock())
    with pytest.raises(RuntimeError, match="reports 'STATE -'"):
        cap.SerialLights(stream=FakeSerial(deaf), settle_s=0).set({3})
    with pytest.raises(RuntimeError, match="no reply"):
        cap.SerialLights(stream=FakeSerial(ctl, silent=True), settle_s=0).set({3})
    with pytest.raises(RuntimeError, match="ERR no channel 9"):
        cap.SerialLights(stream=FakeSerial(ctl), settle_s=0).set({9})
    good = cap.SerialLights(stream=FakeSerial(ctl), settle_s=0)
    good.set({1, "backdrop"})
    assert ctl.on == {"1", "backdrop"}
