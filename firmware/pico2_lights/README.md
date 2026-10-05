# Pico 2 light controller

Switches the rig's 8 PS lights and the backdrop for `capture_session.py` (`SerialLights`), over USB serial. Written for the parts recommended in CLAUDE.md (Rig → lights): Yuji COB LEDs on Mean Well **NLDD-1400H** constant-current drivers, one 48 V supply, a Raspberry Pi **Pico 2**. Tested on the PC only (`tests/test_L_lights_firmware.py` runs this `main.py`); the checks below need the hardware.

## How it switches

The NLDD-1400H's DIM input turns the driver **on when open** (or above 2.5 V) and **off when shorted** to its −Vin (below 0.8 V). So each channel has an N-MOSFET across DIM and −Vin, and the Pico drives its gate:

| gate | MOSFET | DIM | light |
|---|---|---|---|
| high (pull-up; also at reset and unpowered) | on | shorted | **off** |
| low (Pico drives it) | off | open | **on** |

A light therefore stays off until the firmware switches it on, even while the Pico boots or if its USB cable is out. Only static on/off is used: DIM's PWM input would chop the LED current, which is the flicker and banding the rig avoids.

## Wiring (per channel)

- Pico GPIO → 100 Ω → MOSFET gate; **4.7 kΩ from the gate to the Pico's 3V3** (pin 36). The RP2350's pins come out of reset as inputs with a weak pull-down, so the external pull-up must win: 4.7 kΩ keeps the gate near 3 V.
- MOSFET drain → the driver's DIM; source → the driver's −Vin.
- **Tie the drivers' −Vin (the 48 V supply's −V) to the Pico's GND**, or the gate has no reference.
- MOSFET: logic-level N-channel, fully on at 2.5 V gate drive (for example AO3400A). Avoid parts that need 4.5 V or more on the gate.
- The backdrop uses the same logic (pin low = on): the same circuit on its own NLDD driver, or a relay board with active-low inputs.

| channel | GPIO | Pico pin |
|---|---|---|
| light 1 | GP2 | 4 |
| light 2 | GP3 | 5 |
| light 3 | GP4 | 6 |
| light 4 | GP5 | 7 |
| light 5 | GP6 | 9 |
| light 6 | GP7 | 10 |
| light 7 | GP8 | 11 |
| light 8 | GP9 | 12 |
| backdrop | GP10 | 14 |

Change `PINS` in `main.py` to use other pins. Label each light with its channel number: `lights.json` (stage C) describes the lights by the same ids.

## Flashing

1. Install MicroPython for the Pico 2 (board `RPI_PICO2`, from micropython.org): hold BOOTSEL, plug in USB, copy the `.uf2` onto the drive that appears.
2. `pip install mpremote`, then `mpremote cp main.py :main.py` and `mpremote reset`.
3. It appears as `/dev/ttyACM0` (check `ls /dev/ttyACM*`); put that in `rig.json` (`lights.port`) and install pyserial in `.venv`.
4. By hand: `python -m serial.tools.miniterm /dev/ttyACM0 115200`, then type `ID?`, `L1=1`, `?`, `ALL=0`. Ctrl-C (which `mpremote` sends) stops the firmware for updates; the host never sends it.

## Protocol

One ASCII line per command, one line back:

| command | effect | reply |
|---|---|---|
| `ALL=0` | all off | `OK` |
| `L<ch>=1`, `L<ch>=0` | channel on / off (`ch`: 1–8, `backdrop`) | `OK` |
| `?` | which channels are on | `STATE 3,backdrop`, or `STATE -` |
| `ID?` | identify | `GWPS-LIGHTS 1 channels=1,...,8,backdrop` |
| `AUTO=<s>` | auto-off after s seconds on (0 = never; default 120) | `OK` |
| anything else | nothing | `ERR <why>` |

`SerialLights` (with `"verify": true` in `rig.json`, the default) reads every reply, raises on anything but `OK`, and after switching asks `?` and raises unless exactly the requested lights are on: a lost or garbled command stops the session instead of lighting a frame wrongly.

**Auto-off:** a light on for more than 120 s switches everything off (the next `STATE` ends in `auto-off`). The capture script sets the lights for every frame, for a few seconds at most, so this only catches a host that stopped mid-session with lights on. Each COB is on for ~6 % of a session; its heatsink is rated for full power anyway.

## Checks once the hardware exists

1. Each channel on and off by hand (`L<ch>=1`), and the right physical light responds; nothing is lit after power-up or with USB unplugged.
2. **Settle time:** the time from `L<ch>=1` to steady light output (driver soft-start); set `lights.settle_s` in `rig.json` above it (0.3 s now).
3. A flat frame at the working shutter shows no banding (it would mean PWM or a noisy supply).
4. The pilot's stage C additivity and drift checks then confirm the controller in use.
