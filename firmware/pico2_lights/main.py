"""Light controller for the gw-ps-diagnostic rig: MicroPython on a Raspberry Pi Pico 2.

Nine channels, the 8 PS lights and the backdrop. Each GPIO drives the gate of an N-MOSFET that shorts
a Mean Well NLDD-1400H's DIM pin to the driver's -Vin: gate high = DIM shorted = light OFF (also the
state from reset, through the gate's pull-up), gate low = DIM open = light ON. Static on/off only:
DIM's PWM input would chop the LED current (flicker, banding). Wiring and flashing: README.md.

Protocol: USB serial, one ASCII line per command; every reply is one line.
  ALL=0          all off                                            -> OK
  L<ch>=1 | =0   one channel on or off (ch: 1-8 or backdrop)        -> OK
  ?              the channels that are on                           -> STATE 3,backdrop   (STATE - if none)
  ID?            identify                                           -> GWPS-LIGHTS 1 channels=1,...,8,backdrop
  AUTO=<s>       switch everything off after s seconds on (0 never) -> OK
  otherwise                                                         -> ERR <why>
Safety: a light on for longer than AUTO_OFF_S (120 s) switches everything off, and the next STATE
ends in "auto-off". The capture script sets the lights for every frame, for seconds at a time, so
this only catches a host that stopped mid-session with lights on.
Host side: gwps.capture.SerialLights reads every reply and checks the STATE (verify=True).

The protocol logic (Controller) is plain Python: the tests run this file's Controller on the PC.
"""
PINS = {"1": 2, "2": 3, "3": 4, "4": 5, "5": 6, "6": 7, "7": 8, "8": 9, "backdrop": 10}   # channel -> GPIO
ON_LEVEL = 0            # gate low = MOSFET off = DIM open = light on
AUTO_OFF_S = 120
VERSION = 1
MAX_LINE = 64


def _order(ch):
    return (0, int(ch), "") if ch.isdigit() else (1, 0, ch)


class Controller:
    """outputs: {channel: set_level(level)}; now_ms() and diff_ms(a, b) give the clock."""

    def __init__(self, outputs, now_ms, diff_ms=lambda a, b: a - b, on_level=ON_LEVEL, auto_off_s=AUTO_OFF_S):
        self.out, self.now_ms, self.diff_ms = outputs, now_ms, diff_ms
        self.on_level = on_level
        self.auto_off_ms = int(auto_off_s * 1000)
        self.on, self.since, self.auto_off_fired = set(), None, False
        self.all_off()

    def _write(self, ch, on):
        self.out[ch](self.on_level if on else 1 - self.on_level)

    def all_off(self):
        for ch in self.out:
            self._write(ch, False)
        self.on, self.since = set(), None

    def switch(self, ch, on):
        self._write(ch, on)
        if on:
            if not self.on:
                self.since = self.now_ms()
            self.on.add(ch)
        else:
            self.on.discard(ch)
            if not self.on:
                self.since = None

    def handle(self, line):
        """One command line -> its reply (None for an empty line)."""
        cmd = line.strip()
        if not cmd:
            return None
        if cmd == "?":
            state = ",".join(sorted(self.on, key=_order)) or "-"
            return "STATE " + state + (" auto-off" if self.auto_off_fired and not self.on else "")
        if cmd == "ID?":
            return "GWPS-LIGHTS %d channels=%s" % (VERSION, ",".join(sorted(self.out, key=_order)))
        if "=" not in cmd:
            return "ERR unknown command " + cmd[:MAX_LINE]
        key, val = cmd.split("=", 1)
        if key == "ALL":
            if val != "0":
                return "ERR ALL takes 0"
            self.all_off()
            self.auto_off_fired = False
            return "OK"
        if key == "AUTO":
            try:
                s = float(val)
            except ValueError:
                return "ERR AUTO takes seconds"
            if s < 0:
                return "ERR AUTO takes seconds >= 0"
            self.auto_off_ms = int(s * 1000)
            return "OK"
        if key.startswith("L") and len(key) > 1:
            ch = key[1:]
            if ch not in self.out:
                return "ERR no channel " + ch[:MAX_LINE]
            if val not in ("0", "1"):
                return "ERR L<ch> takes 0 or 1"
            self.switch(ch, val == "1")
            return "OK"
        return "ERR unknown command " + cmd[:MAX_LINE]

    def tick(self):
        """Call often: the auto-off."""
        if self.on and self.auto_off_ms and self.diff_ms(self.now_ms(), self.since) > self.auto_off_ms:
            self.all_off()
            self.auto_off_fired = True


def main():
    import select
    import sys
    import time
    import machine
    pins = {ch: machine.Pin(gp, machine.Pin.OUT, value=1 - ON_LEVEL) for ch, gp in PINS.items()}   # off at once
    led = machine.Pin("LED", machine.Pin.OUT, value=0)
    ctl = Controller({ch: p.value for ch, p in pins.items()}, time.ticks_ms, time.ticks_diff)
    poll = select.poll()
    poll.register(sys.stdin, select.POLLIN)
    buf = ""
    while True:
        if poll.poll(20):
            c = sys.stdin.read(1)
            if c in ("\n", "\r"):
                reply = ctl.handle(buf)
                buf = ""
                if reply is not None:
                    sys.stdout.write(reply + "\n")
            elif len(buf) < MAX_LINE:
                buf += c
        ctl.tick()
        led.value(1 if ctl.on else 0)


if __name__ == "__main__":
    main()
