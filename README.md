# gw-ps-diagnostic

Decides, from evidence, whether a garment scan's **Gaussian Wrapping (GW)** mesh is right at mid and low surface scales (s ≥ 5 mm). It compares the mesh with **photometric-stereo (PS)** normals from a fixed multi-light rig, and gives one of these verdicts:

| Verdict | Meaning | Next step |
|---|---|---|
| **BAKE** | GW shape is right at s ≥ 5 mm; PS adds only fine detail | `stage_albedo.py` + `stage_bake.py` → `.glb` |
| **BUILD** | GW is wrong at s ≥ 5 mm in a way PS can see and that is not PS bias | justify training-time PS supervision (L_PS) |
| **INCONCLUSIVE** | PS on this fabric and rig is too biased (fabric floor > 3°) | fix PS first |
| **UNEXPLAINED** | clusters exist, but they rotate with the lights (PS bias) | report, don't build |

`CLAUDE.md` is the single source of truth: conventions, data contracts, decisions, measured numbers and the dated status log. This README only covers getting started on a **freshly installed Fedora 44** PC.

---

## 1. System packages (Fedora 44)

```bash
sudo dnf install -y git python3.11 python3.11-devel gcc gcc-c++ time
```

- **Python 3.11.** The project's reference environment is Python 3.11 (`stage0_env.py` needs ≥ 3.10). Fedora 44's default `python3` is newer, and binary wheels for `embreex`, `pycolmap` and `rawpy` may lag behind it, so use the `python3.11` package.
- **`time`** (GNU time, `/usr/bin/time`) is needed by `src/bench_mesh_size.py`, `src/gw_pod.sh` and their tests (`test_Z_mesh_size.py`, `test_W_gw_dryrun.py`). Fresh installs often lack it.
- `gcc` / `python3.11-devel` are only a fallback in case pip has to build a package from source.

Optional, only when the hardware arrives (see §7):

```bash
sudo dnf install -y gphoto2        # camera tethering (libgphoto2 2.5.33)
sudo dnf install -y colmap         # only if you want the colmap CLI; the pipeline uses pycolmap
```

No NVIDIA driver or CUDA is needed on this machine. The whole diagnostic runs on CPU; only GW training needs a GPU (rented pod, §6).

---

## 2. Clone

```bash
git clone git@github.com:dawud-tan/gw-ps-diagnostic.git
cd gw-ps-diagnostic
```

The repository is public: add your SSH key to GitHub first (`ssh-keygen -t ed25519`, then paste `~/.ssh/id_ed25519.pub` under GitHub → Settings → SSH keys).

Git ignores `.venv/`, `runs/` and `captures/`, so a fresh clone has none of them. `tests/conftest.py` creates `runs/`.

---

## 3. Python environment

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip wheel

.venv/bin/pip install \
  "numpy==1.26.4" \
  scipy \
  "trimesh" embreex \
  "opencv-python-headless==4.11.0.86" \
  "rawpy==0.27.1" \
  "pycolmap==4.2.0" \
  pillow \
  "colour-science==0.4.6" \
  xatlas fast_simplification \
  pytest

# optional: serial light controller / turntable (only with the hardware)
.venv/bin/pip install pyserial
```

Why the pins:

| Package | Pin | Reason |
|---|---|---|
| `numpy` | 1.26.4 | `colour-science` 0.4.7 pulls in numpy 2 (CLAUDE.md → Gotchas) |
| `colour-science` | 0.4.6 | same; used for ColorChecker colour correction (`gwps/albedo.py`) |
| `pycolmap` | 4.2.0 | the API notes in CLAUDE.md (methods vs properties, mask naming, `Sim3d`) are for 4.2 |
| `opencv-python-headless` | 4.11 | ChArUco pixel-convention and `phaseCorrelate` findings were measured on 4.11 |
| `rawpy` | 0.27.1 | LibRaw 0.22.1; RAW decoding of the camera body |
| `embreex` | — | **required**: trimesh's pure-Python ray fallback runs out of memory |
| `xatlas`, `fast_simplification` | — | UV atlas and decimation in `stage_bake.py` |

### Check the environment (stage 0)

```bash
PYTHONPATH=src .venv/bin/python src/stage0_env.py
```

Every line must read `ok`, including `ok   embree RayMeshIntersector constructed and casting`. A `FAIL embreex` means ray casting would silently fall back to trimesh's slow path. Fix it before going on.

---

## 4. Run the tests

```bash
.venv/bin/python -m pytest tests -s
```

- **Expected result** (last full run, 2026-10-01): **130 passed, 1 xfailed.** The xfail is the documented vertical-bias blind spot (`test_vertical_row_dependent_bias_is_not_rejected`); the pipeline flags such clusters with `vertical_warning`.
- **Time:** ~45 min on an 8-core CPU, longer on 4 cores.
- **Disk:** ~2 GB of synthetic data under `runs/_pytest`. `pytest.ini` sets `--basetemp=runs/_pytest` because Fedora's `/tmp` is a RAM-backed tmpfs with a per-user quota; pytest clears that directory at the start of each run.

Quicker subsets:

```bash
# the eight core synthetic tests (~3 min)
.venv/bin/python -m pytest tests/test_[1-8]_*.py -s

# one area, e.g. capture + firmware (no hardware needed: fake gphoto2 / fake serial)
.venv/bin/python -m pytest tests/test_K_capture.py tests/test_L_lights_firmware.py -s

# skip the minutes-long tests
.venv/bin/python -m pytest tests -m "not slow"
```

Test file prefixes:

| Prefix | Covers |
|---|---|
| `test_1`…`test_8` | the eight acceptance tests in CLAUDE.md (PS accuracy, no false alarms, bump found, ripple ignored, light-fixed bias rejected, near-light, parsing, projection) |
| `test_C*` | stage C: ChArUco board, intrinsics, ball centres, light positions, intensity, pilot fabric floor, distorted-lens pilot |
| `test_R_*` | radiometry (noise model, linearity, additivity, drift), model error, 24 MP memory |
| `test_U_*`, `test_G_*` | lit-image undistortion; stage 1 / stage 2 gates |
| `test_K_*`, `test_L_*` | capture sessions (mock + fake gphoto2), light-controller firmware |
| `test_M_*`, `test_S_*` | masks; COLMAP chain on a synthetic garment session |
| `test_V_*`, `test_H_*`, `test_I_*` | consistency bins, board row coverage, cluster height in mm |
| `test_Y_*`, `test_Z_*` | garment selection; 2M-face mesh-size benchmark |
| `test_A_*`, `test_N_*` | albedo / colour correction; bake to `.glb` (incl. size budget) |
| `test_D_*` | dark fabrics and exposure brackets |
| `test_W_*` | GW dry-run kit, the whole post-capture chain with stand-in GW meshes |
| `test_B_*` | ChArUco board generator |

---

## 5. A first synthetic run by hand

Every stage is a small CLI script in `src/` with `--help`. Library code lives in `src/gwps/`. Always run with `PYTHONPATH=src`.

```bash
export PYTHONPATH=src
PY=.venv/bin/python

# list each stage's arguments
for s in src/stage*.py src/synth_*.py src/undistort_lit.py src/capture_session.py; do
  echo "== $s"; $PY "$s" --help | head -20
done
```

Generate a synthetic dataset in the real file formats (COLMAP model, manifest, lights, lit images):

```bash
$PY src/synth_make.py --out runs/demo_bump  --kind torso --variant bump  --noise
$PY src/synth_make.py --out runs/demo_board --kind board                --noise
```

`--variant` picks what is wrong with the mesh or PS: `bump` (GW lost a 3 mm bump → expect BUILD), `ripple` (fine weave only → expect BAKE), `bias` (light-fixed PS bias → expect UNEXPLAINED), `vbias` (vertical blind spot).

Then run stages 3 → 7 in order (check each script's `--help` for its exact inputs; the tests in `tests/test_3_bump_found.py` and `tests/test_5_bias_rejected.py` show complete invocations):

| Stage | Script | Does |
|---|---|---|
| 3 | `stage3_mesh_maps.py` | embree ray casting: hit mask, depth, position, face id, normal per view |
| 4 | `stage4_ps.py` | near-light PS solve per pixel, with per-light shadow/saturation rejection |
| 5 | `stage5_compare.py` | vector low-pass at s ∈ {1, 2, 5, 10, 20, 50} mm, per-face θ̄ and consistency |
| 6 | `stage6_floor.py` / `stage6_controls.py` | fabric floor from the flat-board control |
| 7 | `stage7_verdict.py` | clusters, rules 1–4 → `verdict.md` + JSON |

The whole post-capture chain on a synthetic garment session (masks → COLMAP → stage 1 → stages 2–7 → bake) is scripted in the GW dry-run kit:

```bash
$PY src/gw_dryrun.py prepare --out runs/gw_dryrun --bump     # ~10 min at 1920 × 1440
```

`finish` needs a GW mesh back from a GPU; see §6.

---

## 6. GW on a rented GPU (not on this PC)

GW (commit `11e3b6f`, radegs variant) needs CUDA. Rent a pod, don't buy yet:

- **GPU:** 48 GB RTX A6000 or L40S (runs GW's pinned torch 2.3.1 / CUDA 12.1 as is; RTX 50-series needs torch ≥ 2.7 / CUDA 12.8).
- **Image:** CUDA 12.1 **devel** (needs `nvcc`). **Host RAM ≥ 32 GB** (training images stay in host RAM), disk ~50 GB.

```bash
# on the pod, after uploading gw_dataset_upload.tar and src/gw_pod.sh
tar xf gw_dataset_upload.tar
bash gw_pod.sh setup
bash gw_pod.sh run gw_dataset gw_out    # -r 1 --no_postprocess; logs VRAM, RAM, time
bash gw_pod.sh pack gw_out              # → gw_out.tgz

# back here
$PY src/gw_dryrun.py finish ...         # see docs/gw_dryrun.md
```

Full walkthrough: `docs/gw_dryrun.md`. A full 30k-iteration run likely exceeds 30 minutes; agree on it before starting a paid run.

---

## 7. Real data: when the rig exists

Order of work (details in `docs/capture_checklist.md` and CLAUDE.md → "Order of work once the rig exists"):

1. **Camera:** `sudo dnf install gphoto2`, `gphoto2 --auto-detect`, one `gphoto2 --capture-image-and-download`.
2. **Config:** `cp config/rig.example.json config/rig.json`, then replace the Canon-style setting names with the body's own (`gphoto2 --list-config`, `--get-config <name>`). Set `ext`, `shutter_setting`, `all_lights_shutter`. `config/rig.json` is per machine.
3. **Lights:** flash `firmware/pico2_lights/main.py` to a Raspberry Pi Pico 2 (wiring, parts and hardware checks in `firmware/pico2_lights/README.md`), `.venv/bin/pip install pyserial`. On Fedora, add yourself to `dialout` for `/dev/ttyACM0`: `sudo usermod -aG dialout $USER`, then log out and in.
4. **ChArUco board:** `.venv/bin/python bikin_papan_charuco.py --help` designs and prints a board for the camera and distance; it writes a vector PDF and `board.json`.
5. **Pilot session:** `capture_session.py --plan pilot`, then stage C:
   - `stageC_radiometry.py`: noise model, linearity, additivity, LED drift → `radiometry.json`
   - `stageC_intrinsics.py`: OPENCV intrinsics from ~15 ChArUco shots
   - `stageC_lights.py`: light positions (mirror + matte ball), E / axis / μ → `lights.json`
   - `undistort_lit.py`, then `stage6_controls.py`: fabric floor per placement
6. **Garment session:** `capture_session.py --plan garment`, then `--develop`.
7. `stage_masks.py` → `stage_sfm.py --intrinsics …` (COLMAP, turntable frame in metres, gates) → `stage1_gw_prep.py --masks …` → GW on the pod (§6).
8. `stage2_frame_gate.py` → `stage3_mesh_maps.py` → `stage_garment.py` → stages 4–7 → `runs/<name>/verdict.md`.
9. On **BAKE**: `stage_albedo.py apply` → `stage_bake.py` (50k faces, `.glb` ≤ 15 MB).

Never move, refocus or zoom the camera after the pilot: the lights are stored in the camera's frame.

---

## 8. Repository layout

```
CLAUDE.md                 spec, conventions, decisions, status log (read this next)
README.md                 this file
pytest.ini                --basetemp=runs/_pytest
bikin_papan_charuco.py    camera-aware ChArUco board generator
config/rig.example.json   camera / lights / turntable / capture plans template
src/                      one CLI script per stage
src/gwps/                 library: io, ps, compare, capture, sfm, masks, garment, bake, synth, …
src/gw_pod.sh             GW setup / run / pack on a GPU pod
tests/                    pytest suite (synthetic data, fake cameras, fake serial)
firmware/pico2_lights/    MicroPython light controller for a Pico 2
docs/                     capture checklist, GW dry run, prior-art report, evidence, refs
drafts/                   earlier attempts with verified bugs; reference only, don't extend
runs/                     (ignored) outputs and pytest scratch
captures/                 (ignored) raw captures
```

Outputs land in `runs/<name>/<stage>/`: per-view `.npz` (float32, camera frame), a `meta.json` per stage, per-face heatmap `.ply` files, JSON and Markdown summaries.

---

## 9. Troubleshooting on Fedora

| Symptom | Fix |
|---|---|
| `pip` builds `embreex` / `pycolmap` / `rawpy` from source and fails | you are on Fedora's default Python; recreate `.venv` with `python3.11` |
| `FAIL embree ray casting` in stage 0 | `.venv/bin/pip install --force-reinstall embreex` |
| every test errors at setup with a missing `runs/...` directory | run pytest from the repo root (`tests/conftest.py` creates `runs/`) |
| `test_Z_mesh_size` / `test_W_gw_dryrun::test_pod_script_run_and_pack` fail with `/usr/bin/time: not found` | `sudo dnf install time` |
| `No space left on device` mid-suite | free ~5 GB; the suite writes ~2 GB under `runs/_pytest`, the dry-run kit more |
| ImportError mentioning numpy 2 after installing extra packages | something upgraded numpy; `pip install "numpy==1.26.4" "colour-science==0.4.6"` |
| `ModuleNotFoundError: gwps` | prefix commands with `PYTHONPATH=src` |
| `/dev/ttyACM0: Permission denied` | `sudo usermod -aG dialout $USER`, log out and back in |
| `gphoto2: Could not claim the USB device` | a desktop auto-mounter (gvfs) grabbed the camera: `gio mount -s gphoto2`, or `pkill -f gvfs-gphoto2` |
| repo on an NTFS drive: a directory briefly refuses deletion | wait and retry; NTFS over FUSE can lag after writers exit |
