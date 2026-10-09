# CLAUDE.md: Gaussian Wrapping × photometric-stereo disagreement diagnostic

## Goal
Decide, with evidence, which of three verdicts applies to a garment scan.

- **BAKE:** Gaussian Wrapping's (GW) mesh shape is right at mid and low surface scales (s ≥ 5 mm, see Conventions). Photometric-stereo (PS) normals add only fine detail, so bake them into a UV normal map on the decimated mesh and stop.
- **BUILD:** GW's mesh is wrong at those scales, in a way the PS normals can see and that isn't PS bias. That justifies building training-time PS supervision (L_PS).
- **INCONCLUSIVE:** PS on this fabric and rig is too biased to judge. Fix PS first; in that case even baking is questionable.

Background and prior art are in `docs/prior_art_report.md`. As of 2026-09-24 no published work supervises GW with PS normals; the report sketches an L_PS design. **Implementing L_PS is out of scope until there is a verdict.**

## Status (dated log; last updated 2026-10-01)
- **Current state:** see the 2026-09-27, 2026-09-28, 2026-09-29 and 2026-10-01 bullets at the end of this list. Earlier bullets are history, so some of their "not built" lines are out of date. The next section lists what waits on hardware and the order of work once the rig exists.
- `drafts/` holds earlier attempts from three different generations. They contradict each other and contain verified bugs (listed below; repro scripts in `docs/evidence/`). Use them as reference only. Don't extend them in place.
- `drafts/CLAUDE.previous.md` mentions `photometric_stereo.py`, `gw_ps_diagnostic.py` and `synth_test/`. **They are not in this directory, and Dawud no longer has them (answered 2026-09-25).** Everything is written fresh under `src/` and `tests/`.
- **Stages 0, 3, 4, 5, 6 (synthetic controls) and 7 are built** in `src/`, with the library in `src/gwps/`. All 8 synthetic tests pass in both noise settings (19 passed plus 1 documented xfail; about 3 min on CPU).
  - Env: `.venv/` (Python 3.11): `PYTHONPATH=src .venv/bin/python src/stage0_env.py`, then `.venv/bin/python -m pytest tests -s`.
  - `src/synth_make.py` writes a synthetic dataset in the real formats.
  - Stages C, 1 and 2 aren't built. Real-image undistortion (Conventions) isn't built either, because the synthetic camera is already PINHOLE.
- **2026-09-26: stage C (intrinsics, ball centres, light positions, E / axis / μ) and the stage 6 control builders (ChArUco board poses, matte-ball centres) are built** and tested on a synthetic pilot capture (`src/synth_pilot.py`; tests `tests/test_C*.py`). Scripts: `src/stageC_intrinsics.py`, `src/stageC_lights.py`, `src/stage6_controls.py`.
  - Full suite (8 original tests plus C1–C6): about 10 min on CPU; pytest writes to `runs/_pytest` (see Gotchas).
  - Not built yet: the metric-scale check, the camera-response calibration (only needed without RAW), a fit of the PS noise model from dark and flat frames (stage 4 still uses the synthetic defaults), and stages 1 and 2.
- **2026-09-26: lit-image undistortion (`src/undistort_lit.py`, `src/gwps/undistort.py`), the pilot capture checklist (`docs/capture_checklist.md`) and a camera-aware ChArUco generator (`bikin_papan_charuco.py`, adapted from `/WIN_D/protek/charuco/bikin_papan_charuco.py`) are built and tested** (`tests/test_U_*.py`, `tests/test_C7_*.py`, `tests/test_B_*.py`).
  - The whole real-data pilot flow runs on a synthetic pilot rendered through a distorted lens: `stageC_intrinsics` → `stageC_lights` (raw images) → `undistort_lit` → `stage6_controls`. Fabric floor 0.08–0.10° at s ≥ 5 mm.
  - Pilot processing is ready; what's missing is real captures (see the checklist).

- **2026-09-27: everything that can be built before the rig exists is built and tested on synthetic data.**
  - **Real-data robustness of PS:**
    - `stageC_radiometry.py` fits the noise model (read noise, full well) and checks linearity, additivity and LED drift; stages C, 4 and 6 take the result with `--noise`.
    - A model-error term and relative confidence.
    - Chunked stages 3–5: a 27 MP view peaks at ~4.5 GB per stage.
  - **`capture_session.py`:** pilot and garment sessions in the checklist's order, writing the manifests (this defines the file naming).
  - **`stage1_gw_prep.py`:** principal-point gate and identical crop.
  - **`stage2_frame_gate.py`:** reprojection, depth, lit-image alignment and overlays.
  - **`stage_albedo.py`:** ColorChecker colour correction and per-channel albedo.
  - **`stage_bake.py`:** decimation, UV atlas, baked normal and base-colour textures, `.glb` export.
  - Tests: `tests/test_R_*`, `test_K_*`, `test_G_*`, `test_A_*`, `test_N_*`. **Full suite: 72 passed plus 1 documented xfail (the vertical-bias blind spot) in 17 min 28 s on CPU** (`runs/pytest_full.log`; re-run 2026-09-27 after the all-lights shutter added 6 capture tests).
  - Not built, but buildable now:
    - the SfM/alignment step and the metric-scale check (next section);
    - camera-response calibration (only needed without RAW).
  - Blocked on hardware (next section):
    - the GW run;
    - the camera-specific gphoto2 tests;
    - the light-controller firmware.
- **2026-09-27 (afternoon): web research recorded under Rig, the GW item below, Data contracts and Fabric suitability.**
  - Covers the light parts, mount, second camera, GPU (rent first), `.glb` delivery limits and fabric tiers.
  - **Found and fixed the same day:** at the PS exposure, all-lights frames (SfM, flats, intrinsics) clip on light fabrics. `capture.py` now gives them their own shutter, `all_lights_shutter` (see Rig → lights).
  - **Found:** the working shutter will be ~0.4–3 s, not 1/60 s. `rig.example.json` now uses 0.5 s for the PS shutter and 1/15 s for the all-lights shutter.
  - **Camera news:** the R8 Mark II (announced 2026-09-15) adds IBIS.
- **2026-09-27: git.** The project is a git repository, pushed to `git@github.com:dawud-tan/gw-ps-diagnostic.git` (public; branch `main`). `.gitignore` excludes `.venv/`, `runs/`, `captures/`, Python caches and `docs/business_strategy.md`.
- **2026-09-28: pre-rig audit. Not everything left waits on the rig;** the new items are under "Not built yet" in the next section.
  - **Rule 3 gives a false BUILD at the planned step angle.** The synthetic tests turn 12 × 30°; the garment plan turns 36 × 10°. At 36 × 10° the camera-frame bias of test 5 scores corrected MRL 0.711 (noisy) / 0.709 (noise-free) at s = 50 mm, against 0.638 / 0.636 at 30° steps. With ~12 views per face instead of ~4, the (n − 1) correction shrinks toward the raw R̄ (0.74), and the correction was what kept it below 0.7. Logs: `runs/steps_experiment/`.
  - **Fix (adopted 2026-09-28, Dawud):** average each face's unit d_s within 30° turntable bins, then take the corrected MRL over the bins (see Decisions → consistency statistic).
  - **Correction:** the GW memory estimate below had assumed the training images sit in VRAM; the radegs wrapper keeps them in host RAM.
- **2026-09-28 (afternoon): the verdict fix, object masks and the COLMAP-to-stage-1 chain are built and tested on synthetic data** (Dawud asked for all three).
  - **Verdict:** `CompareParams.consistency_bin_deg = 30`; `compare.consistency_bins` groups views by the camera centre's azimuth about world +Z; stage 5 visits views grouped by bin; `faces.npz` gains `n_bins_<s>` and `consistency_bin_deg`; stage 7 corrects by bins. Tests 3 and 5 now run at 12 × 30° and 36 × 10°. At 36 × 10° the bias scores 0.612 / 0.614 (noise-free / noisy, s = 50 mm; 0.711 per view on the same data) and the bump 0.94 / 0.88 and 0.92 / 0.85 (s = 5 / 10 mm); at 12 × 30° the numbers are unchanged. Unit tests: `tests/test_V_consistency_bins.py`.
  - **Masks:** `gwps/masks.py`, `src/stage_masks.py`. The garment plan now shoots a backlit **silhouette** frame per step (`mask_source`, default `silhouette`; `sfm` = threshold the SfM image on a matte black backdrop; `none`). A static exclusion PNG removes the turntable's base. Synthetic IoU against the true masks: 0.989 (silhouette), 0.989 (black backdrop, SfM image); without the exclusion the base leaks in (IoU 0.971). Tests: `tests/test_M_masks.py`.
  - **COLMAP chain:** `gwps/sfm.py`, `src/stage_sfm.py`: masked SIFT, exhaustive matching with COLMAP's stationary-match filter, incremental mapping with the pilot's intrinsics fixed; the turntable frame from the circle of camera centres; the scale from a one-axis bundle fit to the metric-board frames; gates; `image_undistorter`; masks and lit frames undistorted with the same mapping. Stage 1 writes RGBA (alpha = mask); stage 2 strips the alpha and draws the mask. Synthetic garment session (`gwps/synth_sfm.py`: 36 × 10° with 1° of jitter, distorted lens, textured static studio, 1920 × 1440): 36/36 registered, 0.26 px; camera-to-axis 1.5998 m against 1.6000; camera centres 0.66 mm RMS and rotations 0.027° RMS from the truth; stage 2 with the true mesh passes (depth 0.15 mm, 0.08 % of mesh pixels outside the masks). Tests: `tests/test_S_sfm_chain.py`, updated `test_K_capture.py`.
  - **Found:** with a textured static background and COLMAP's defaults (no masks, stationary filter off), **COLMAP registered no model**. The stationary filter alone (no masks) rescued it: 36/36, centres 0.62 mm RMS. Masks were run only together with the filter; GW has no such filter, so masks stay.
  - **Found:** COLMAP self-calibration (no pilot intrinsics) puts the scale **2.5 % off** with SIMPLE_RADIAL: it fixes the principal point at the image centre (this lens's is 30 px off) and cannot model k2 or tangential terms (focal 3104 against 3000 px). The board-axis gate catches it (`test_self_calibration_is_never_silently_wrong`). `stage_sfm.py` now requires `--intrinsics` and allows self-calibration only as an explicit `--self-calibrate MODEL` for dry runs.
  - **Found:** the old metric-board plan (4 frames over a full turn) left a flat board edge-on or back-facing in half the frames; the plan now shoots it facing the camera at −40…40°.
  - **Full suite: 92 passed plus 1 documented xfail (the vertical-bias blind spot) in 30 min 19 s on CPU** (`runs/pytest_full.log`). The verdict tests at 36 × 10° and the COLMAP chain add ~13 min.
  - Page updated the same day (status line, backdrop, masks and COLMAP, the 10° fix, the not-built list).
- **2026-09-28 (evening): garment selection, the vertical-bias flag and the mesh-size test are built** (Dawud asked for all three).
  - **Garment selection:** `gwps/garment.py`, `src/stage_garment.py` → `garment_faces.npy` (+ `.ply` with the garment red, `.json`). Criteria: a region cut in the turntable frame; a **bare-mannequin reference** (the same mannequin scanned once without a garment), registered and then faces within 2 mm dropped; optional per-view garment masks voted through stage 3's face-id maps; small pieces dropped. Synthetic dressed mannequin (`gwps/synth_garment.py`): region cut alone precision 0.74 / recall 1.00; with the reference moved by 25° and 3.9 mm, registration left 0.38° and 0.07 mm, precision 1.000 / recall 0.992; unregistered 0.87 / 0.90; four views of eroded masks: labelled faces 100 % right. Tests: `tests/test_Y_garment.py`.
  - **Found (registration):** a loose inlier score picks wrong angles, because the reference's body, hidden under the garment, crosses the garment when misrotated (at 5 mm, no rotation scored 0.582 against 0.566 for the truth); and a tight score of rotation alone is noise, because a few mm of offset pushes the shared surfaces past it (0.28 at 2 mm for the true angle, 0.53 with its offset). So every 2° start gets a short trimmed ICP (rotation and offset) before scoring, and the best four peaks are refined.
  - **Vertical-bias flag:** stage 5 writes each face's mean resultant vector (`dmean_<s>`); stage 7 reports each cluster's vertical share d_z²/|d|² and, when a BUILD rests on clusters with ≥ 0.8, `vertical_warning` and a warning in `verdict.md`. Measured: the bump 0.53–0.57 (no warning, at 12 × 30° and 36 × 10°), the vertical blind-spot bias 0.98–1.00 (all 8 BUILD clusters flagged), the camera-frame bias 0.69–0.71. A genuinely missing horizontal fold would read as vertical too: suspicion, not rejection. Test: `test_vertical_build_is_flagged`.
  - **Bake target:** `stage_bake.py --target-faces 50000` (default; `--keep` for a ratio). The old ratio of 0.3 would have left millions of faces from a GW mesh, against Scene Viewer's 100k. Synthetic bake unchanged in quality (135,320 → 50,000 faces; normals 0.53° median, ΔE00 0.61).
  - **Concavity flag scaling:** its neighbourhoods are now drawn from ≤ 200k area-weighted face centres (on a 10M-face garment a 0.1 m ball holds ~500k faces, ~10⁹ neighbours over 2000 queries).
  - **Mesh size** (`src/bench_mesh_size.py`; each stage in its own process under `/usr/bin/time`): the synthetic torso with the bump, 4 views at 30° steps.
    - **2M faces at 1920 × 1440** (`tests/test_Z_mesh_size.py`, ~3 min): every stage ≤ 2.1 GB and ≤ 33 s; the bump found (MRL 0.96 / 0.94 at s = 5 / 10 mm); bake 2,000,336 → 50,000 faces.
    - **10M faces at 6000 × 4500 (24 MP)**, by hand: rendering 417 s; stage 3 79 s / 4.5 GB; stage 4 342 s / 3.8 GB; stage 5 326 s / **6.8 GB** (9.7 GB before the fixes below); stage 6 (board at 24 MP, 5 placements) 469 s / 4.8 GB; stage 7 17 s / 5.7 GB; bake 41 s / 6.7 GB (9,998,450 → 50,000 faces); `stage_garment.py` 13 s / 4.3 GB (region cut; registration adds ~8 s). The bump found (0.96 / 0.95). Garment faces seen by some view: 57 % (4 views).
    - **A 36-view garment at 24 MP** therefore takes ~2 h of CPU on this 8-core machine (stages 4 and 5 ~85 s per view each, stage 3 ~20 s), plus the fabric floor once per fabric; peak memory stays ~7 GB, since views are processed one at a time.
    - **Fixed on the way:** stage 5 pooled every valid pixel's θ for its summary (~200 MB per 24 MP view, ~7 GB for 36 views; now 200k per view and scale); stages 5 and 6 kept the previous view's ~2 GB of inputs and outputs alive while comparing the next (now released); the face accumulator used float64 sums and n_faces-long temporaries per view and scale (now float32 / int16 and the view's own faces via `np.unique`, which is also faster). Results unchanged.
    - **Still large:** one 24 MP view's comparison peaks at ~4.5 GB (full-frame low-pass buffers); cropping to the garment's bounding box would roughly halve it but shifts the block grid of the large-σ low-pass slightly, so it is not done.
  - **Full suite: 100 passed plus 1 documented xfail (the vertical-bias blind spot, now flagged by `vertical_warning`) in 30 min 32 s on CPU** (`runs/pytest_full.log`).
  - Page updated the same evening (status line, garment selection, the vertical flag, the 10M-face benchmark, the 50k-face bake, what is left before the rig).
- **2026-09-29: the remaining pre-rig items** (Dawud: "complete remaining open items that could be done without the full rig").
  - **GW dry-run kit** (`src/gw_dryrun.py`, `src/gw_pod.sh`, `docs/gw_dryrun.md`): `prepare` renders a synthetic garment session with PS frames (`make_session(ps=True, bump=..., scale=...)`: 8 single-light frames per step with cast shadows) and runs masks → stage_sfm → stage 1 → the fabric floor, writing a ~50 MB upload (1920 × 1440; `--scale 3.125` for 24 MP). `gw_pod.sh setup | run | pack` installs GW at 11e3b6f on the pod (checking that it imports: its install.py ignores failed builds), runs it at `-r 1 --no_postprocess` while logging VRAM, host RAM and time, and packs the raw mesh and logs. `finish` runs stage 2, stage 3, garment selection (region cut), stages 4, 5, 7 and the bake on the returned mesh and measures its signed distance to the true surface. **Everything but the GPU step is tested here** (`tests/test_W_gw_dryrun.py`, ~11 min): the pod script against stand-ins for conda, nvcc, nvidia-smi and GW; and prepare → finish with stand-ins for GW's mesh, a 3 mm bump on the garment: the mesh without the bump → BUILD, bump flagged (MRL 0.91 / 0.84 at s = 5 / 10 mm, vertical share 0.54–0.56), max 2.92 mm from the true surface; the true surface → BAKE, max 0.12 mm; stage 2 passes both; bake 126,480 → 50,000 faces, coverage 1.000. This is also the first run of the whole real-data chain after capture with PS: masks, COLMAP, stage 1, stages 2–7, the bake.
  - **Not done: the GPU run itself.** It needs Dawud to rent a pod (48 GB A6000 or L40S, CUDA 12.1 devel image, ≥ 32 GB RAM) and to OK a run over 30 minutes; then three commands (`docs/gw_dryrun.md`).
  - **Light-controller firmware:** built for the recommended parts, tested on the PC (item 2 of the next section).
  - **Customer interviews:** a guide added to `docs/business_strategy.md` (section 9, private); holding them is Dawud's.
  - **Found:** on `/WIN_D` (NTFS over FUSE), a directory could refuse deletion for a moment after the processes writing into it were killed; `gw_pod.sh run` now waits for its loggers.
  - **Full suite: 109 passed plus 1 documented xfail (the vertical-bias blind spot, flagged by `vertical_warning`) in 43 min 29 s on CPU** (`runs/pytest_full.log`). The dry-run kit's test adds ~11 min.
- **2026-10-01: five pre-rig items that needed neither hardware nor a decision** (Dawud: "do them in the order 1, 2, 3, 4, 5", from a list of what can be built while the GPU is funded). Done in a cloud session on a fresh clone (4 cores).
  1. **Dark garments: exposure brackets** (`ps_brackets` in the garment plan, `gwps.io.merge_brackets`, the exposure-aware noise model, stage 2's bracket check) and **`src/bench_dark_fabric.py`**, which measures PS error and the fabric floor against albedo at the planned pixel density. Result: with a quiet sensor the verdict copes with near-black fabric, and brackets are for the baked normals (1.43° → 0.33° at albedo 0.03); with the suite's noisier sensor a board at albedo 0.03 is INCONCLUSIVE at one exposure (floor 3.57° at s = 5 mm) and fine bracketed (0.041°). Numbers and the selection-bias fix under Decisions → exposure brackets. Tests: `tests/test_D_dark_fabric.py`, `test_K_capture.py::test_garment_exposure_brackets`.
  2. **Board coverage of the blind spot** (stage 5 `elev`, stage 6 `<kind>_placements`, stage 7 `row_coverage` and each cluster's floor at its own height). It reports and leaves the rules alone. Decisions → known blind spot; tests `tests/test_H_row_coverage.py`.
  3. **Implied height of a cluster in mm** (`gwps/height.py`; stage 7 with `--sparse --stage3 --stage4`; also in `gw_dryrun.py finish`): the missing 3 mm bump reads +2.62 mm at 2 mm, +2.15 mm at s = 5 mm. Masked least squares rather than Frankot–Chellappa (FC was off by 0.32 on a 0.5-high bump over an irregular region), and only views within 45° of face-on (beyond, the low-passed field's curl was 8 % of its steps and the integrators disagreed by 1.6 mm). Pipeline step 7; tests `tests/test_I_height.py`, `test_W_gw_dryrun.py`.
  4. **The `.glb` size budget** (`stage_bake.py --max-mb`, JPEG base colour when needed, normal map always PNG). The default 2048² synthetic bake measured 9.06 MB, under the limit. Data contracts → delivery limits; test `tests/test_N_bake.py`.
  5. **Prior-art gaps, partly**: the GW code is unchanged at 11e3b6f; web search found no citing work that supervises GW with PS normals (five new 2026 papers listed, none close); the GitHub issues, PRs and forks are still unread (attaching the repository with Dawud's GitHub credentials was refused by the session's permission check) and no citation index was reachable (egress policy). `docs/prior_art_report.md` → "Gap check (2026-10-01)".
  - **Found:** a fresh clone could not run a single test (no `runs/` for `--basetemp`); `tests/conftest.py` now creates it. **Found:** `docs/business_strategy.md` is tracked in git despite `.gitignore` (see the summary page bullet).
  - **Full suite: 130 passed plus 1 documented xfail (the vertical-bias blind spot)**: the 109 + 1 of 2026-09-29 and 21 new tests. Run on the 4-core cloud container in two halves (14 min and 30 min; the logs stayed in that container, which is not kept), the second partly alongside the dark-fabric benchmark. Five of half B's tests (in `test_W`, `test_Y`, `test_Z`) failed only because the session's disk allowance ran out (`No space left on device`); after clearing scratch data, those three files passed on a rerun (10 passed in 19 min). The unchanged code passed the same way here once GNU time was installed (Gotchas).

## Blocked until the rig is provisioned (recorded 2026-09-27)
Everything that runs on synthetic data is built and passes (2026-09-29); what is left before the rig needs Dawud ("Left before the rig" below). The numbered items wait on hardware or on the first real capture. A fresh session should start here, and should also confirm the open ⚠ items under Rig with Dawud.

**Not installed on this machine (2026-09-27):**
- gphoto2 / libgphoto2;
- the `colmap` CLI;
- pyserial in `.venv`;
- an NVIDIA driver (`nvidia-smi` is absent).

pycolmap 4.2.0 and rawpy 0.27.1 (LibRaw 0.22.1) are in `.venv`.

1. **Camera-specific gphoto2 tests (blocked: no body yet).**
   - **Built:** `Gphoto2Camera` in `src/gwps/capture.py`.
     - It sets each setting with `gphoto2 --set-config`, reads it back with `--get-config`, and captures with `--capture-image-and-download --filename … --force-overwrite`.
     - `Session.lock` raises if a setting reads back differently from what was set.
     - It has been tested only against a fake `gphoto2` script (`tests/test_K_capture.py::test_gphoto2_driver`), never against a real body.
   - **When the body arrives:**
     1. Install gphoto2 (`sudo dnf install gphoto2`; Fedora 44 ships libgphoto2 2.5.33). Run `gphoto2 --auto-detect`, then one `--capture-image-and-download`.
     2. Copy `config/rig.example.json` to `config/rig.json`.
        - Its setting names and values (`imageformat`, `iso`, `aperture`, `shutterspeed`, `whitebalance`, `capturetarget`) are Canon-style guesses. Replace them with the body's own, from `gphoto2 --list-config` and `--get-config <name>`.
        - Also set `ext` (`cr3` for Canon, `nef` for Nikon) and the pilot plan's `shutter_setting`.
     3. Check that rawpy decodes the body's RAW files. A deliberately overexposed frame must reach ≥ 0.98 of full scale in `gwps.io.load_linear`, which is the saturation threshold.
     4. Time one capture-and-download, and one shutter change with its read-back. A garment session in `rig.example.json` is 36 steps × 11 frames (SfM, silhouette, ambient and 8 lights) with 72 shutter changes, plus 5 metric-board frames.
     5. Dry-run `capture_session.py --plan pilot` with manual lights before the real pilot.
   - The settings to turn off, and the f/8 against f/11 test, are in the camera ⚠ item under Rig.
2. **Light-controller firmware: written for the recommended parts (2026-09-29, `firmware/pico2_lights/`); what remains needs the hardware.**
   - **Built:** `SerialLights` in `src/gwps/capture.py` sends one text line per command over serial.
     - It sends `ALL=0`, then `L{id}=1` for each light to switch on, then waits `settle_s` (0.3 s). The strings are configurable in `rig.json`.
     - Channel names beyond the PS ids, such as `backdrop`, are allowed.
     - It has been tested only against a byte stream (`test_serial_lights_protocol`).
     - `ManualLights` prompts the operator instead.
   - **Built (2026-09-29):** MicroPython `firmware/pico2_lights/main.py` for a Pico 2 (9 channels: GP2–GP9 = lights 1–8, GP10 = backdrop). Gate low = light on (an N-MOSFET across each NLDD-1400H's DIM, gate pulled up, so every light is off from reset and with USB out). Replies to every command (`OK` / `ERR …`), `?` → `STATE 3,backdrop`, `ID?`, and an auto-off after 120 s on. `SerialLights` now reads every reply and checks the state after switching (`"verify": true` in `rig.json`; false for a controller that never replies). Wiring, parts, flashing and the hardware checks: `firmware/pico2_lights/README.md`. Tested on the PC (`tests/test_L_lights_firmware.py`): the flashed file's protocol, gate levels, auto-off and `main()` wiring against fake `machine`/serial, a whole garment session through a simulated controller (the lights on at each of 23 captures were exactly those asked for), and a controller that ignores a command, answers ERR or stays silent.
   - **Not tested:** on the hardware (the README's checks: each channel by hand, off at power-up, settle time, banding).
   - **When the hardware is chosen:**
     - Install pyserial in `.venv`.
     - Make the firmware accept the protocol, or change the strings.
     - Keep the drivers constant-current (no PWM).
     - Map channel ids to the physical lights that `lights.json` describes.
     - Measure the settle time, and check a flat frame for banding at the working shutter speed.
     - Stage C's additivity and drift checks then confirm the controller.
   - **Turntable driver:** it has the same shape. `SerialTurntable` sends `ROT {deg:.3f}` and waits 2 s. `rig.example.json` uses the manual turntable, where the operator rotates it and confirms.
3. **GW training (blocked: needs the RTX server).**
   - **Built:** `stage1_gw_prep.py` writes the GW dataset, and writes the exact command into `stage1.json`: `train_and_extract_gw_radegs.py -s <dataset> -m <gw_out> --no_postprocess -r 1`.
   - **On the RTX server:**
     - Install GW at commit 11e3b6f in its own conda env. An RTX 50-series (Blackwell) card needs a newer torch/CUDA than the pinned 2.3.1; see Verified facts.
     - **Ask Dawud before the run:** 30k iterations plus mesh extraction will likely exceed the 30-minute threshold.
     - Copy back the raw `…_searched.ply`, and record the `-r` used.
   - **Which GPU (recommendation, 2026-09-27): rent first.**
     - **Prices:** GPUs cost ~2× MSRP in September 2026: RTX 5090 ~US$4,300+ against its US$1,999 MSRP; RTX PRO 6000 96 GB US$14–16k; a used RTX 4090 ~US$2,500–3,000.
     - **What to rent:** a 48 GB Ampere/Ada card runs GW's pinned torch 2.3.1 / CUDA 12.1 without the Blackwell rebuild. RTX 50-series cards need torch ≥ 2.7 with CUDA 12.8. RunPod: RTX A6000 48 GB US$0.33–0.53/h, L40S 48 GB US$0.79–1.09/h, RTX 5090 32 GB US$0.69–0.99/h.
     - **Measure** peak VRAM and time on the first run.
     - **Memory at `-r 1` (corrected 2026-09-28):** the radegs wrapper trains and extracts with `--data_device cpu`, so the float32 training images (36 × 24 MP ≈ 10.4 GB; ≈ 13.8 GB with an alpha mask) sit in **host RAM** and are copied to the GPU per iteration. VRAM goes to the Gaussians (≤ 6M, `--N_max_gaussians`) and the 24 MP render buffers: measure it. Rent a pod with ≥ 32 GB of RAM. The 48 GB Ampere/Ada choice still stands because of the pinned torch, not because of image memory.
     - **Time (estimate):** the GW paper reports 27 min on Tanks and Temples (GPU not stated). 24 MP is ~15× the pixels of the 1600 px default, so expect hours.
     - **Buying:** only once throughput is known. A US$4,300 5090 equals ~4,300–6,300 rented 5090-hours.
   - Then run `stage2_frame_gate.py`, stages 3–7, and, on a BAKE verdict, `stage_bake.py`.
   - **Dry-run kit (2026-09-29):** `docs/gw_dryrun.md`. `gw_dryrun.py prepare` (here) → `gw_pod.sh setup / run / pack` (pod) → `gw_dryrun.py finish` (here). For a real garment the same `gw_pod.sh run` takes stage 1's dataset.

**Built 2026-09-28 (were on this list):** SfM, alignment, the metric-scale check, `image_undistorter` (all in `stage_sfm.py`), rule 3 at 36 × 10° (30° bins), background masks (`stage_masks.py`), garment face selection (`stage_garment.py`), the vertical-bias flag, and the mesh-size test (`bench_mesh_size.py`, `tests/test_Z_mesh_size.py`). See the 2026-09-28 status bullets.
- ⚠ **Mask source: built both, default `silhouette`** (a backlit frame per step, +36 frames and ~2 min per garment; works for any garment colour). The alternative is a matte black backdrop (`"mask_source": "sfm"`), which fails on dark garments. Segmentation is not built. Dawud confirms the backdrop when the rig is built (see Rig).

**Left before the rig (2026-09-29); each needs Dawud:**
- **The GW dry run on a rented GPU:** the kit is ready and tested here (`docs/gw_dryrun.md`); it needs a pod and an OK for a run over 30 minutes.
- **The firmware's hardware checks** (`firmware/pico2_lights/README.md`), once the parts are bought.
- **Customer interviews** (guide: `docs/business_strategy.md` section 9).
- **One decision from 2026-10-01, no build needed:** whether rule 3 should hold a cluster against the floor at its own height (stage 7 reports it) instead of the pooled floor. (The other, whether `docs/business_strategy.md` should stay tracked in git, is settled: the history was replaced by one commit on 2026-10-06 that doesn't contain it; see the summary page bullet.)
- **The prior-art leftovers** (`docs/prior_art_report.md` → "Gap check"): read GaussianWrapping's issues, PRs and forks in a browser or with `gh`, and Google Scholar's "cited by", neither of which the cloud session could reach.
- **Not needed unless decided.** Neither needs hardware to build, only a decision (2026-09-29); both can be tested on synthetic data and fake cameras like the rest.
  - **Camera-response calibration:** only if the chosen body can't deliver RAW that rawpy decodes. The recommended bodies shoot RAW, and item 1 checks the decoding when the body arrives. A synthetic test would only recover a tone curve we invented: in-camera JPEGs are 8-bit, sharpened, noise-reduced and possibly locally tone-mapped, which one response curve can't undo, so only stage C's linearity and additivity checks on the real camera could judge it.
  - **A second camera:** more than a camera loop in `capture_session.py`.
    - The file names (`steps/stepNNNN/<light_id>`, `stepNNNN.png`) would collide.
    - `stage_masks.py` takes one static exclusion for all views.
    - `stage_sfm.py` runs COLMAP with one shared set of intrinsics (`CameraMode.SINGLE`) and fits one circle of camera centres.
    - Stage 4 takes one noise model, and stage 7 one fabric floor; each camera has its own.
    - Already per camera: stage C, stage 4's lights, stage 6, stages 1–2 and the undistortion.
    - The 30° consistency bins group views by turntable angle, so both cameras' views of a step would share a bin: one sample per angle, presumably the conservative choice. A synthetic two-camera test should confirm it.

**Order of work once the rig exists:**
1. Pilot session (`docs/capture_checklist.md`).
2. Stage C and stage 6: the processing chain in the checklist.
3. Decide on cross-polarisation from the crossed and parallel fabric floors.
4. Garment session (checklist → garment session), then `capture_session.py --develop`.
5. `stage_masks.py`, then `stage_sfm.py` (SfM, turntable frame in metres, gates, undistorted images, masks and lit frames).
6. `stage1_gw_prep.py --masks …`, then GW on the RTX server.
7. Stage 2, stage 3, `stage_garment.py` (with the bare-mannequin reference, scanned once per mannequin), then stages 4–7 and the verdict.
8. On a BAKE verdict: `stage_albedo.py apply`, then `stage_bake.py`.

**Published summary page:**
- URL: https://dawud-tan.github.io/garment-scan-camera-ps-gltf.html (GitHub Pages; moved from the earlier self-hosted `http://183.81.158.231:8080/…`, checked 2026-10-09).
- Source: `/opt/src/dawud-tan.github.io/garment-scan-camera-ps-gltf.html`, with an entry in that repository's `sitemap.xml`.
- 2026-10-09: the page links to this public repository (header, section 11 `#code`, JSON-LD `isBasedOn`) and states the last full test run (130 passed plus 1 xfail); also the firmware's replies and auto-off, and the open rule-3 floor decision.
- It restates this file's recommendations and numbers as of 2026-09-27, plus the 2026-10-01 additions (exposure brackets for dark garments and their measured need, the board-coverage check, the cluster height in mm, the measured 9.1 MB bake and the JPEG fallback), plus the 2026-09-28 additions: the GPU memory correction, the 10° step issue and its adopted fix (section 11, `#step-angle`), masks and the COLMAP chain as built, garment selection, the vertical-bias flag, the 10M-face / 24 MP benchmark, the 50k-face bake, and what is left before the rig (`#not-built`: the GW dry run, whose kit is ready and needs a pod, and the firmware, written and waiting for its hardware checks; 2026-09-29; the status line says the chain is built). When a decision here changes, update the page, its JSON-LD dates and the sitemap `lastmod`.
- Dawud commits and pushes the page himself (2026-09-27; GitHub Pages since then): edit the source, don't commit or push it.
- **JSON-LD: hardware goes in `mentions` as `Thing`, never `Product`** (2026-09-28). Google's Product markup needs `offers`, `review` or `aggregateRating`, and the page neither sells nor rates anything. Search Console flagged all 10 `Product` mentions as invalid.
- **Pricing and business strategy stay off the page** (Dawud, 2026-09-27). They were kept in `docs/business_strategy.md` (private notes: offers, price hypotheses, unit economics, risks, validation plan), which is listed in `.gitignore`. It had been committed before that (`07d09bc`, `87e040a`; found 2026-10-01). **Resolved by 2026-10-09:** the repository's history is now a single commit (`166d00e`, "initial commit", 2026-10-06) that doesn't contain the file, and the public GitHub repository shows only that commit. The file lives only on Dawud's development PC (`docs/business_strategy.md` in his working copy there, git-ignored) and is kept off the public repository (Dawud, 2026-10-09), so a checkout elsewhere, such as this machine or a cloud clone, won't have it. Never commit it, and don't link to it or quote it on the page.

## Decisions and deviations (2026-09-25, from the synthetic tests; review them)
- **Consistency statistic (step 5 and rule 3):**
  - Rule 3 uses the *view-count-corrected* mean resultant length, R̃ = √(max(0, (n R̄² − 1)/(n − 1))). The raw R̄ sits at chance at 0.67 / 0.54 / 0.47 / 0.42 for n = 2–5 views, so 0.7 barely clears chance.
  - R̃ is taken only over faces whose views span ≥ 50° of azimuth about the turntable axis. Adjacent 30° steps see nearly the same smooth light-fixed bias, so 2-view faces scored raw 0.92 on a pure camera-frame bias.
  - A cluster with no such faces has consistency "n/a" and can't be a BUILD candidate. `verdict.md` also reports the raw R̄.
  - Measured on the synthetic scene: the bump scores 0.92 / 0.88 at s = 5 / 10 mm (raw 0.94 / 0.92). The camera-frame bias scores 0.36–0.64 (raw 0.56–0.76). **The margin at s = 50 mm is thin (0.64 against 0.7).**
  - **Samples are 30° turntable bins, not views (2026-09-28).** n counts views as independent, but views a few degrees apart see nearly the same light-fixed bias: at 36 × 10° the bias scored 0.711 at s = 50 mm, a false BUILD. So each face's per-view unit d_s are averaged within 30° bins of camera azimuth about +Z (centred on the first image's), and R̃ is taken over bins, with n = bins (≥ 2 needed). At 30° steps this is exactly the old statistic. At 36 × 10°: bias 0.61, bump 0.92 / 0.85 (noisy). The model must be in the turntable frame (stage_sfm), since bins use world +Z. `CompareParams.consistency_bin_deg` (0 = per view).
- **⚠ Known blind spot:** a light-fixed bias that tilts normals *vertically* as a function of image row looks garment-fixed, because turntable steps never move a face in v. The synthetic test gives R̃ = 1.00 and a false BUILD (`test_vertical_row_dependent_bias_is_not_rejected`, xfail).
  - Mitigations: controls at several heights covering the garment's image rows, so the floor sees the bias; a second camera height; or lights at mixed heights.
  - Treat any BUILD whose d_s is mostly vertical with suspicion. **Flagged automatically since 2026-09-28:** a cluster's vertical share d_z²/|d|² ≥ 0.8 puts `vertical_warning` on the BUILD (bump 0.53–0.57, this bias 0.98–1.00).
  - **Board coverage is checked automatically since 2026-10-01.** Stage 6 records each board placement's elevation span in the camera (Y/Z of its valid pixels, 1st–99th percentile) and its own floor; stage 5 records each face's mean elevation; stage 7 reports the share of the garment's area inside some placement's span (`row_coverage`, flagged below 95 %), the uncovered elevations, and per cluster its elevation, how much of it a placement covered, and the highest floor among the placements at its height. A vertical BUILD cluster that no placement covered is called out ("the blind spot is unchecked there; shoot the board at those heights"), and one that is covered but does not exceed its height's floor + 1° is called out as explained by that floor. **It reports; the rules are unchanged.** Whether rule 3 should use the floor at a cluster's height instead of the pooled floor is Dawud's call (the pooled p95 can hide a bias that only one placement saw).
  - Elevation, not the pixel row: it survives stage 1's crop and is the same physical quantity in the controls' camera and the garment's.
  - Measured (synthetic, `vbias_edge`: the same 4° row bias, but only outside the middle ±110 px, where a mid-height board cannot see it): boards at mid height only → BUILD, all 8 vertical clusters "at rows no board placement covered", coverage 45 % (uncovered −12.5…−5.5° and 5.6…12.7°); boards also 15 cm higher and lower (rendered with the same bias, as a real rig would show it) → their floors read 4.07° / 4.08° at s = 5 mm, the pooled floor rises past 3° and the verdict is INCONCLUSIVE, coverage 91 %. Tests: `tests/test_H_row_coverage.py`.
- **Low-pass weights (step 5):** w = apodise(valid) × confidence. The weights taper to 0 over 1.5 px at the valid-mask edge. Hard edges leaked the 3 px ripple into s = 5 mm (p99 0.97° within 5 px of the edge, against 0.02° beyond 10 px).
- **Confidence (step 4):** 1 / max(1, χ²_red), with χ² pooled over a Gaussian of σ = 2 px. Per-pixel χ²_red has std ≈ 0.6 at 5 dof, and as a weight it unbalanced the ripple (face max at s = 5 mm was 0.67°, now 0.38°). Pixels with 3 lights get 0.25; min_conf is 0.3.
- **Controls (step 6) don't remove the per-view Kabsch rotation.** On a board a rotation can't be told from a uniform sheen bias, and the floor must include that bias.
- **Test 4's θ̄ < 0.5° bound applies at s ≥ 5 mm.** At s = 1 mm a 3 px ripple is only partly suppressed (≈ 2.1° remains), as expected.
- **PS light rejection per pixel:** a light is dropped if saturated, below 3σ, or below 0.1 × the pixel's brightest shading. Then up to 2 passes drop the worst light if the fit predicts ≤ 0 for it or its |residual| > 5σ, keeping ≥ 3 lights.
- **Where the MRL test has power:** only for disagreement directions with a horizontal (turntable-rotating) component. More views per face shrink R̃'s chance spread but don't add angular baseline.
- **PS light selection is two-pass (2026-09-26).** Choosing lights per pixel from that pixel's own noisy intensity is a selection bias: a light near the threshold is kept exactly where noise pushed it up, and the normal tilts toward it. Measured: a uniform 0.11° bias on the board at −50° tilt, where one light was kept in ~44 % of pixels, and a fabric floor that plateaued at ~0.115° from s = 5 to 50 mm. Pass 2 selects lights from the predicted shading of a pass-1 normal smoothed over σ = 3 px, and detects cast shadows per pixel only by a residual below −5σ. Afterwards the true-light floor falls with scale as noise should: 0.071 / 0.039 / 0.023 / 0.014° at s = 5 / 10 / 20 / 50 mm.
- **Large-σ low-pass (2026-09-26):** for σ > 8 px, stage 5 anti-alias blurs (σ = f/2), block-averages by f ≈ σ/4, blurs, and bilinear-upsamples, with the variances summing to σ². It is ~11× faster at σ = 100 px (s = 50 mm on 1920 px; worse on 24 MP) and matches the direct blur on the 3 px ripple (s = 20 mm p99 0.0052° against 0.0051°). **Without the anti-alias step** the ripple aliased to DC: p99 0.31°.
- **Noise model from the flat stack (2026-09-27).**
  - RAW decoders subtract the black level and clip at 0, which leaves dark frames with ~0.34 of the read-noise variance (synthetic: 0.0027 measured for a true 0.005).
  - So read noise is the intercept of the flat stack's photon-transfer line, using bin **means**, not medians: the median of a 10-frame sample variance is ~7 % low.
  - Dark frames are used only if unclipped. Recovered: read 0.00499 against 0.005, full well 9,986 against 10,000.
- **Model error and confidence (2026-09-27).**
  - With a quiet, real-sensor-like noise model and a sheen fabric, the old absolute-χ² confidence left **38 %** of the garment usable.
  - Confidence is now χ² relative to the view's median (stage 4 records the median in its metadata), which keeps 100 %.
  - The model-error term ε = 0.005 enters only outlier tests and χ², never the weights. Measured PS error on the sheen fabric: 1.22° at ε = 0, 1.46° at 0.005, 1.66° at 0.02. ε tolerates the rig's ~1–2 % calibration error while specular spikes stay above 5σ.
- **Memory at 24 MP (2026-09-27).** Stage 3 casts in 1M-ray row blocks into float32 maps. Stage 4 solves in 400k-pixel chunks, with light vectors computed per chunk. Stage 5 keeps per-scale θ and d only at valid pixels, and stage 6 subsamples to 2M θ per view and scale. Measured on a 6000 × 4500 view: 4.5 GB peak per stage; 12 s / 81 s / 70 s for stages 3 / 4 / 5.
- **Lit-image alignment gate (2026-09-27).**
  - `cv2.phaseCorrelate` (OpenCV 4.11) reports 0.5 px for identical images at widths such as 616 or 617, and is biased by 0.1–0.2 px elsewhere. So registration uses ECC on shading-normalised images.
  - The lit **set** is checked against the SfM image through the sum of the lit frames: ±0.01 px, gate 0.2 px.
  - Each light is checked against the sum of the other unflagged frames, flagging and removing the worst iteratively: single-light shading bias ±0.15 px, gate 0.3 px, so single-frame bumps are caught from ~0.4 px.
- **Albedo (2026-09-27).**
  - Synthetic chart (camera with channel mixing and gains): leave-one-out ΔE2000 median 0.32 (linear) and 0.23 (root-polynomial); the fitted linear matrix × camera mixing = identity to ±0.003.
  - Garment: noise-free median 0.10; noisy per-pixel 1.46, which is noise: 5 × 5-averaged 0.30.
- **Bake (2026-09-27).** On a ridged garment (±8°, 12 mm cords, visible only to PS): baked normals median error **0.53°** (p90 1.05°) against 5.67° for the decimated mesh alone; base colour ΔE2000 median 0.61; texel coverage 99.99 % from 12 views; decimated 135k → 40.6k faces (since 2026-09-28 the default is 50k faces: 0.53° and ΔE00 0.61 unchanged). Tangents are exported, so viewers don't substitute MikkTSpace ones.
- **Angles are computed in float64 with atan2(|a×b|, a·b).** float32 arccos quantises near 0 in steps of 0.0198°·√k and had been flattening every θ below ~0.2°.
- **Exposure brackets (2026-10-01).**
  - **Merge:** per light and pixel, the longest exposure that does not clip, in units of the PS shutter; the noise model takes the exposure factor k (read variance / k², shot variance / k). **Which frame a pixel takes is decided from the next-shorter frame's smoothed local peak** (Gaussian σ = 2 px, then a 5 × 5 max) times the exposure ratio, below 0.9 of full scale. Deciding from the pixel's own base value is a selection bias (the base is kept exactly where noise pushed it up): on a bright synthetic board it raised the 50 mm floor from 0.022° to 0.038°, where the smoothed decision lowered it to 0.019°. A uniform patch at the switch point: merged mean within 3e-4 of the truth, the own-pixel rule +0.004 (1.7 %).
  - **Alignment gate:** each bracket frame against its own light's base frame scaled by the exposure ratio and clipped where the bracket clips. The each-against-the-others check of the base frames flagged five lights for one bumped 4× frame of the bright torso (a quarter of it clipped), and an ECC mask over the clipped regions gave 2.5 px on clean frames.
  - **Need** (`src/bench_dark_fabric.py`: the fabric board, 5 placements, a crop camera at 5.75 px/mm = 50 mm lens, 6 µm pixels, 1.45 m; exposure anchored so a 90 % white facing the median light reads 60 % of full scale; noise read 1.5e-4, full well 60,000, a guess at a 14-bit full-frame body; ~1 h on 4 cores; the tables below are its output, the run directory was not kept):

    | albedo | PS error median, one exposure / ×4 ×16 brackets | floor at s = 5 mm, one exposure / brackets |
    |---|---|---|
    | 0.5 | 0.32° / 0.21° | 0.008° / 0.006° |
    | 0.12 | 0.67° / 0.21° | 0.018° / 0.005° |
    | 0.03 | 1.43° / 0.33° | 0.040° / 0.008° |
    | 0.008 | 3.24° / 0.65° | 0.227° / 0.017° |

    Every fabric pixel stayed usable (confidence ≥ 0.3) with ~7.8 lights. **With the suite's noisier sensor** (read 0.005, full well 10,000; same density, `--noise test`):

    | albedo | PS error median, one exposure / brackets | floor at s = 5 mm, one exposure / brackets |
    |---|---|---|
    | 0.5 | 1.33° / 0.71° | 0.039° / 0.030° |
    | 0.12 | 4.41° / 0.72° | 0.834° / 0.025° |
    | 0.03 | 18.04° / 1.37° | **3.567° (INCONCLUSIVE)** / 0.041° |

    At albedo 0.03 and one exposure only 82 % of the fabric stayed usable (6.3 lights). At the suite's own pixel density (1000 px focal) a board at albedo 0.031 had a floor of 4.47° at one exposure and 0.21° bracketed (`tests/test_D_dark_fabric.py`). So with a quiet sensor brackets are for the baked normal map's quality, and with a noisy one for the verdict itself; the pilot's measured noise model decides which (`bench_dark_fabric.py --noise radiometry.json`, ~1 h on 4 cores for four albedos).

## Rig (⚠ = confirm with Dawud before touching real data)
- A garment on a mannequin sits on a 360° turntable. Each turntable step captures one image per light, one ambient (all-lights-off) frame, the image used for SfM/GW and (default) a backlit silhouette frame for the object mask.
- ⚠ **Backdrop (2026-09-28):** a backdrop behind the whole garment, lit from behind and switchable (the `backdrop` channel), for the per-step silhouette; or a matte black backdrop with `mask_source: sfm` (fails on dark garments). Plus a static exclusion PNG over the turntable's base and anything else static in front of it. Without masks and with COLMAP's defaults a textured static background broke SfM outright (synthetic), and GW would model it as floaters.
- COLMAP treats the garment as fixed, so the camera appears to orbit it. `stage_sfm.py` puts the model in metres in the turntable frame: origin on the axis at the turntable top (from the tape-measured camera height), +Z up the axis (the circle of camera centres), step 0's camera on −Y, and the scale from the metric-board frames (2026-09-28; `model_aligner` is not used).
- ⚠ **Where the lights are mounted.** *Not decided yet (Dawud, 2026-09-25).* The code and synthetic tests assume camera and lights are both fixed in the studio; the world-frame path stays supported. If the lights ride on the turntable, store them in the world frame instead (see Conventions).
  - **Consequence to weigh before deciding:** the verdict's consistency test (mean resultant length of d_s) relies on PS bias rotating relative to the garment across steps. If the lights ride on the turntable, a light-fixed bias stays fixed on the garment and looks exactly like a shape error, so step 7 rule 3 can no longer separate them. Fixed-in-studio lights are strongly preferred.
  - **Recommended mount** (web research, 2026-09-27): one rigid aluminium T-slot frame (40 × 40 mm profile; ~Rp 900,000 per 6 m in one Tokopedia listing, and Misumi Indonesia sells it there too) carrying the camera(s) **and** all lights. Lights are stored in the camera's frame, so the two must never move relative to each other; on one frame a bump moves both together. No light stands on castors.
- ⚠ Number of physical cameras. Each camera needs its own light calibration.
  - **Optional second camera** (recommendation, 2026-09-27): the same body and lens as camera 1, on the same frame ~0.5–0.8 m higher and tilted down (≥ ~20° more elevation).
    - **Gains:** better GW coverage of shoulders and hems. It also catches row-dependent biases tied to one camera: its own light calibration, and view-dependent sheen. A bias from a light-model error shared by both calibrations stays garment-fixed; only the multi-height board controls bound that one.
    - **Costs:** a second light calibration and its own fabric controls. In software, `capture_session.py` drives one camera, so it needs a camera loop per shot (`Gphoto2Camera(port=…)`), and `stage_sfm.py` and a few other places assume one camera (list under "Left before the rig"; none of it needs the hardware to build). Re-run `gphoto2 --auto-detect` every session, because USB addresses change on replug.
    - **Polarisers:** per-light linear tuning is per viewpoint (Ma et al.), so it can't be crossed for both cameras. Options: circular polarisers; Ghosh et al. 2011's latitude/longitude pattern (cameras must sit near its equator); or tune for camera 1 and measure camera 2's leakage in the pilot.
- Number of PS lights: **6–8, prefer 8** (Dawud, 2026-09-25; exact count and positions still to be calibrated). `drafts/lights.example.json` (3 PS lights plus 1 `diffuse_reference`) does not describe the rig.
  - A suggestion of 4–6 lights (2026-09-27) was rejected. With 8 lights and the tested noise the median PS error is already 0.81° of a 1° budget, and per-pixel shadow rejection needs ≥ 4 lights left, so with 4–6 lights the folds drop to 3 (low confidence).
  - Placement: 30–45° off the camera axis, spread evenly in azimuth (the synthetic rig uses 30–40°). The angle to the surface normal can't be set, since normals vary over the garment. Drive the LEDs with constant current (no PWM flicker).
  - **Beam wide enough that the whole garment volume sits in the flat part of the beam** (≥ ~60° at 1.3 m). A narrow spot, such as the ~15° MR16s in Wu et al. 2025 (built for an A4 sample at 45 cm), covers only ~34 cm at 1.3 m. Its flat-top, steep-edged profile is also poorly described by the cos^μ falloff model. High colour fidelity: CRI > 95 (Wu et al. used 6000 K, CRI > 95).
  - **Recommended parts** (web research, 2026-09-27; list prices that day):
    - **LED:** Yuji **125H** flip-chip COB, 5600 K, CRI 97 typ (95 min).
      - LES Ø 12.5 mm (0.55° at 1.3 m), 120° emission.
      - Rated 100 W at 3 A / 33–39 V, 7,700 lm; US$109 each.
      - Run it at 1.4 A: about half power, cooler, ~3,500–4,000 lm (estimated). Buy 8 plus spares from one bin.
      - A bare 120° COB is close to Lambertian (μ ≈ 1): at ±21° (the garment edge at 1.3 m) it still gives 0.93 of on-axis, smoothly, which the cos^μ model fits.
      - Cheaper alternative: Yuji **400L** (50 W, 1.5 A, 28–31 V, 3,190–3,740 lm, CRI 95+, 120°; US$98 per 2).
    - **Driver:** Mean Well **NLDD-1400H**, the successor of the obsolete LDD-1500H: 10–56 V in, 6–46 V out, 1.4 A; US$5.55.
      - DIM: on at > 2.5–5 V **or open**, off at < 0.8 V or short. Use it only as a static on/off; its PWM input chops the output.
      - Because open means on, make off the default at reset, e.g. an N-MOSFET per channel pulling DIM to −Vin with its gate pulled up. The firmware drives the gate low to switch a light on.
    - **Board:** Raspberry Pi **Pico 2** (US$5): 3.3 V logic, inside DIM's 2.5–5 V high range, and USB CDC, so it shows as `/dev/ttyACM0`. Or an Arduino UNO R4 Minima (US$20), whose 5 V sits at DIM's upper limit. 9 channels: 8 PS lights plus the backdrop.
    - **Power:** one 48 V supply of ≥ 500 W, since all 8 on draw ~8 × 52 W (e.g. Mean Well HRP-600-48).
    - **Heat:** a passive heatsink per COB rated for its full power, so no fan shakes the camera's frame. Each light is on for only ~6 % of a garment session.
    - **Polariser film:** mount it a few cm in front of the COB, not on it.
  - **Light budget** (bare COB, on-axis at 1.3 m, ISO 100; exposure set so a 90 % white facing the light reaches ~60 % of full scale):
    - 4,000 lm gives 753 lx. At f/11 that is 0.40 s, or **3.1 s cross-polarised**; at f/8, 0.21 s or 1.6 s.
    - So the working shutter will be ~0.4–3 s. The earlier 1/60 s came from Wu et al.'s 15° spots at 45 cm. `rig.example.json` now uses this budget without polarisers: 0.5 s for the PS shutter and 1/15 s for the all-lights shutter, with the sweep bracketing 1/15 s. Replace them with the pilot's measurements.
    - A garment session then takes ~30–45 min: ~400 frames (with silhouettes and the metric board) at ~3 s of gphoto2 overhead each (to be measured).
  - **Frames with all 8 lights on** (SfM, flats, intrinsics, metric board) are 2.5–2.8 stops brighter than a single-light frame at 30–45° (Σ cos over 8 lights).
    - At the PS exposure they clip on anything lighter than ~15–25 % albedo, and clipped SfM frames would degrade GW exactly where the verdict then blames GW.
    - **Built (2026-09-27): `camera.all_lights_shutter` in `rig.json`**, ~3 stops (8×) shorter than the PS shutter (`settings[shutter_setting]`).
      - `Session.shoot` uses it for any frame lit by **more than two PS lights**: the SfM and metric-board frames, the flats and the intrinsics shots. Everything else uses the PS shutter: single lights, ambient, dark, silhouette, and the 1+2 pair, which additivity needs at the PS exposure. The sweep sets its own shutters.
      - Each change is read back from the camera, like `Session.lock`. Every capture event in `session.json` logs its shutter, and `session.json` records `shutter: {setting, ps, all_lights}`.
      - A garment session makes 2 changes per step. Downstream is unaffected: gate 2c's ECC is gain-free, and the noise, linearity and additivity fits don't depend on the flats' shutter.
      - Tests: `tests/test_K_capture.py` (mock and fake-gphoto2 sessions, plus the example config run end to end).
- Light type: **bare LEDs, about 1–1.5 m from the garment** (Dawud, 2026-09-25), so `kind: "point"` with an axis and falloff μ. ⚠ The LED axis and μ are still unknown; fit them in stage C.
- ⚠ RAW availability, whether exposure and white balance are locked, and whether ambient frames exist.
- ⚠ **Camera body and lens: not chosen yet; Dawud decides.** Under review (2026-09-27): a mirrorless body (Canon EOS R / Nikon Z) tethered with gphoto2, RAW only, 24 MP, a prime lens. First the recommended option, then requirements that hold whichever body is chosen.
  - **Recommended option** (web research, 2026-09-27): **Canon EOS R8** (24.2 MP full frame, 6000 × 4000, no in-body stabilisation, electronic-first-curtain or fully electronic shutter; $1,499 at launch) with a **50 mm lens that has a mechanical focus ring**, taped. On Canon that means a manual-focus EF lens on Canon's EF-to-RF adapter (no EF model vetted), or a native manual RF lens (below).
    - **2026-09-15: Canon announced the EOS R8 Mark II:** 24.2 MP, but it **adds 5-axis IBIS**; US$1,899, shipping late October 2026; not in libgphoto2 2.5.33. The original R8 is still sold (Rp 20,999,000 body only at an Indonesian dealer, 2026). If you choose the R8, buy both bodies (see the second camera) while stock lasts.
    - **Native RF lens:** TTArtisan 50 mm f/1.4 ASPH (RF). It is fully manual, with no electronic contacts, so tape both rings. The body can't set or report its aperture, so drop `aperture` from `rig.json`'s settings, or `Session.lock` fails its read-back. Check corner sharpness at f/11 in the pilot's intrinsics shots.
    - **Alternative:** Nikon **Z6 III or Z5** with in-body stabilisation permanently off, plus the **Voigtländer APO-Lanthar 50 mm f/2 (Z mount)**: manual focus (reviews give 135° and 160° of throw), US$1,049. Tape the aperture ring too; it sits 4 mm from the focus ring. There is no RF-mount version.
    - **gphoto2:** libgphoto2 **2.5.33** (Fedora 44's package; not installed yet) lists the R8, RP, R6 Mark II, R50, R100, Z5, Z6 II, Z6 III, Zf and A7 IV with capture and preview. The R6 Mark III is only in the development branch; the R5 Mark II and Z5 II are in neither.
    - **Why no in-body stabilisation:** Nocerino, Menna & Verhoeven (ISPRS Archives XLVI-2/W1-2022, 395–400) found it worsened photogrammetric accuracy by up to 300 %. Switching it moved a Z7 II's principal point by ~110 px. Switched off, the bodies calibrated normally, so on a body that has it, never toggle it.
    - **Why a mechanical focus ring:** Nikon Z resets focus-by-wire lenses to infinity at power-off by default. Its "Save focus position" restores them, but Nikon warns the position "may change due to changes in zoom or ambient temperature". Canon's "Retract lens on power off" decides whether some STM lenses keep manual focus.
    - **Full frame over APS-C:** at equal depth of field an APS-C sensor collects less light, and PS here is noise-limited.
    - **To test in the pilot:** 50 mm at ~1.45 m in portrait (35 mm if the room is tight); **f/8** (Unity's photogrammetry-guide default) against **f/11** (needed for ±15 cm of depth); ISO 100; electronic first curtain. Fully electronic shutter only with flicker-free LEDs; R8 users report LED banding with it.
  - Confirm the exact body in libgphoto2's supported list, for remote capture and remote settings.
  - **Turn off** in-body stabilisation, auto power-off, sensor cleaning at power-on/off, in-camera lens corrections, and long-exposure noise reduction (at 0.4–3 s it would add an in-camera dark frame to every RAW and double the capture time). A floating or re-seated sensor shifts the image between shots and breaks per-pixel PS alignment (gate 2c). Between sessions a few-pixel shift is harmless (~0.03° rotation of the calibrated lights), but never power-cycle during one.
  - Electronic shutter only with flicker-free LED drivers; check a flat frame for banding.
  - **Framing** (full frame, 0.9 m garment + margin = 1.0 m on the 36 mm side, portrait): 35 mm → 1.0 m, 50 mm → 1.44 m, 85 mm → 2.45 m, 100 mm → 2.88 m. "50–100 mm at 1–2 m" works only at the 50 mm end. Lens distortion hardly matters: it is calibrated and undistorted (`undistort_lit.py`).
  - **Depth of field is the optical limit**, not megapixels or distortion. At that framing, with pixel-level sharpness (blur ≤ 2 px of 6 µm):

    | aperture | depth of field | diffraction blur |
    |---|---|---|
    | f/8 | ±77 mm | 1.8 px |
    | f/11 | ±106 mm | 2.5 px |
    | f/16 | ±153 mm | 3.6 px |

    The garment surface spans about ±15 cm as it turns, so use f/11 and check the nearest and farthest surfaces. Depth of field depends on framing and aperture, not on focal length. 24 MP is enough: at f/11 more pixels are diffraction-limited, and GW trains at 1600 px wide by default.
  - **Dark garments:** a locked exposure puts them near the noise floor, and PS error grows roughly as 1/albedo. **Built (2026-10-01): exposure brackets.** The garment plan's `ps_brackets` (shutters longer than the PS shutter, e.g. `["2", "8"]` over 0.5 s) reshoots the ambient frame and every PS light at each, every step (9 frames per bracket per step); stage 4 merges each light per pixel to the longest exposure that does not clip and gives the noise model each pixel's exposure factor; stage 2 checks the bracket frames' alignment. **Keep the PS shutter itself at the pilot's**: E and the colour correction were measured there, and albedo comes out in its units (normals do not depend on it). See Decisions → exposure brackets for the measured need.
  - **Linearity** (RAW): an exposure-time sweep on one evenly lit patch, plus additivity (an all-lights frame equals the sum of the ambient-subtracted single-light frames). Rejection in `gwps/ps.py` is noise-based (3σ from the noise model, saturation at 98 % of the raw frame), not a fixed fraction of the range. The `shadow_thresh=8` / `sat_thresh=250` in `drafts/ps_disagreement_check.py` are 8-bit JPEG thresholds.
- ⚠ **Cross-polarisation: recommended, not decided; Dawud decides after the pilot.** Polarising gel on each light, a linear polariser on the lens. It removes specular reflection, the main risk to rule 1 on sheened fabric, and removes false COLMAP matches on highlights.
  - **Recommended option** (web research, 2026-09-27): **linear polarisers, tuned per light.**
    - **Tuning:** Ma et al. (EGSR 2007) tuned "linear polarizers placed over each light source ... to minimize the observed specular reflection from a spherical test object as viewed through the camera's linear polarizer"; the chrome ball does this.
    - **Why it holds for every turntable step:** the tuning depends on the viewpoint, but here camera and lights never move, only the garment does.
    - **Axes:** find each filter's axis with an LCD screen, which emits linearly polarised light (Unity's guide).
    - **Circular alternative:** same-handed circular polarisers on every light and on the lens need no tuning. They separate well within ~70° of the camera and degrade beyond that (Ma et al.).
    - **Separation:** diffuse = 2 × crossed; specular = parallel − crossed (circular: specular = 2 × (parallel − crossed)).
    - **Caveats:** diffuse (cross-polarised) normals are blurred by subsurface scattering, and the effect depends on wavelength (Ma et al.). That softens the baked fine detail more than it affects the s ≥ 5 mm verdict. Specular reflection is weak near the Brewster angle, so specular recovery is unstable there.
    - **Light loss: about 3 stops on the diffuse component.** Wu et al. 2025's filters transmit 42 % (LED film) and 30 % (lens CPL), so 0.42 × 0.30 ≈ 13 %; the extinction ratio is 99.9 %. Plan LED power and exposure for it, and measure it in the pilot.
    - **Don't copy Wu et al.'s pairwise shortcut.** They rotated the lens polariser until one opposite LED pair was darkest and turned the other pairs 90° by eye, which leaves two of four pairs at 45°: only half the lights are crossed. Tune every light.
  - It also removes the mirror-ball highlight that stage C triangulates. Shoot the mirror ball with the lens polariser **parallel** and everything else crossed. Fit E with the gels in place (expect ~2 stops of light lost).
  - **Decide from data:** capture the pilot's fabric board crossed and parallel, and compare the two fabric floors.
- ⚠ **Albedo and colour for the glTF material: built as recommended below (`stage_albedo.py`, `stage_bake.py`, 2026-09-27); waiting for Dawud's sign-off.**
  - **Recommended pipeline** (web research, 2026-09-27):
    1. Decode RAW linearly with rawpy, as now (Unity's guide uses `dcraw -4`, the same idea).
    2. Solve normals on the channel mean with cross-polarised frames, as now; then solve per-channel albedo with the normals fixed. PS albedo here is relative to the primer, because E absorbs its albedo.
    3. **Absolute scale and white balance** from the ColorChecker's neutral 8 patch (luminance factor Y = 58.9 %) or white 9.5 (91.3 %; BabelColor averages), photographed at the garment position under the rig's lights. Published datasets differ by 0.5–1.4 ΔE, so this is about ±2 %.
       - **Match the reference to the chart's age** (checked in colour-science 0.4.6, 2026-09-27). Neutral 8 hardly changes: 58.9 % before X-Rite's November 2014 revision, 59.0 % after. White 9.5 drops from 91.3 % to 88.1 %. `stage_albedo.py calibrate` defaults to the post-2014 data, so a chart made before November 2014 needs `--reference` with `ColorChecker24 - Before November 2014` converted to linear sRGB. Unity's guide does the same by setting its "60 % grey" patch to 0.6. Spectralon is needed only for tighter accuracy.
    4. **Colour:** a 3×3 matrix fitted to all 24 patches with colour-science, `colour.colour_correction(measured, measured, reference)`. The reference is `colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']` converted to linear sRGB, and the input must be linear floats. Detect the chart with `colour-checker-detection`.
       - Use LEDs of one model and bin with high colour fidelity (TM-30 Rf ≥ 90, or CRI ≥ 95 with a good R9). Otherwise fit a matrix per light, since spectra may differ.
    5. **glTF export:**
       - `baseColorTexture` "MUST contain 8-bit values encoded with the sRGB opto-electronic transfer function" (glTF 2.0). glTF's dielectric BRDF weights the diffuse by 1 − F with F ≈ 0.04, so baseColor ≈ cross-polarised albedo / 0.96. Metallic 0.
       - `normalTexture` is tangent space with +X right and +Y up, and comes from the BAKE output (`stage_bake.py`).
    6. **Validate** on colour samples not used for fitting, reporting ΔE00 (JND ≈ 2.3).
       - Optional second stage for accuracy: Wu et al. 2025 went from mean ΔE00 8.20 to 2.29 (39 held-out RAL samples; worst were blues at 4.89) with a ColorChecker profile plus a CIELAB 3D lookup table from 154 RAL samples.
       - Here any second stage must act only on the final **linear** albedo, never on PS inputs. A candidate that keeps exposure scaling intact is the root-polynomial fit (Finlayson 2015, a method of `colour.colour_correction`) on a larger chart such as the ColorChecker SG.
    7. **Albedo comes from the PS solve**, not from an all-lights frame as in Wu et al. (flat samples). On a 3D garment such a frame still contains shading.
    8. **Sheen:** `KHR_materials_sheen` targets "cloth and fabric materials". It adds `sheenColor` (sRGB texture) and `sheenRoughness` (alpha channel) using the "Charlie" BRDF (Conty & Kulla 2017). Start with hand-set values; fitting them needs parallel − crossed frames per light, or roughness has to be hand-set too.
  - **Closest precedent, read 2026-09-27:** Wu, Morosi & Caruso, "Custom Material Scanning System for PBR Texture Acquisition: Hardware Design and Digitisation Workflow", *Applied Sciences* 15(20), 10911 (2025), https://doi.org/10.3390/app152010911 (local copy: `docs/refs/applsci-15-10911.pdf`).
    - **Their rig:** a flat-sample (A4) scanner with 8 relay-switched 12 V DC MR16 LEDs (~15° beam, 6000 K, CRI > 95), a Lumix G80 with a 12–60 mm zoom at 450 mm, RAW, ISO 100, f/11, 1/60 s.
    - **Their method:** cross-polarised; one all-lights frame for albedo and 8 single-light frames for normals; a two-stage colour calibration (ColorChecker DCP via X-Rite software + Lightroom, then the 3D lookup table in MATLAB); reconstruction in Substance 3D Designer.
    - **Confirms:** 8 lights, RAW with everything locked, f/11, high-CRI DC LEDs, cross-polarisation, and ColorChecker profiling. Also that PS "does not directly yield the microfacet roughness parameter": their roughness is heuristic.
    - **Their normals are not a target:** mean error 2.25° (slope) and 1.80° (cylinder), p95 ~3.9°, from Substance's "Multi Angle to Normal" node with a hand-tuned intensity of 0.70, with no calibrated light positions. Our calibrated near-light solve aims under 1°.
    - **Not copied:** the pairwise polariser shortcut, colour transforms before PS, albedo from an all-lights frame, narrow-beam LEDs at garment distance, and a stabilised zoom-lens body.
- File naming: **defined by `src/gwps/capture.py`** (2026-09-27).
  - Calibration frames: `calib/<target>/<position>/<light_id>.<ext>`.
  - Garment frames: `steps/stepNNNN/<light_id>.<ext>`.
  - The SfM/GW image is `steps/stepNNNN/sfm.<raw>`, developed identically to `colmap_images/stepNNNN.png`. It is taken with all PS lights on by default; `sfm_lights` in the rig config changes that.
  - The capture script writes `calib_manifest.csv`, `manifest.csv`, `manifest_parallel.csv` and `session.json`.
  - Per step, the backlit mask frame is `steps/stepNNNN/silhouette.<ext>` (manifest `light_id` `silhouette`), and the garment session's metric-board frames are `calib/metric_board/m<k>/all.<ext>` in its own `calib_manifest.csv` (2026-09-28).
- ⚠ How the camera intrinsics were obtained: COLMAP self-calibration or an external calibration. This decides the principal-point gate below.

## Fabric suitability (2026-09-27; from this pipeline's physics, not yet measured)
- **Good:** matte, opaque, textured fabrics of mid to light colour: cotton jersey, twill and canvas, denim, linen, non-fuzzy wool, knits, cotton batik.
- **Caution; the fabric floor decides:**
  - dark fabrics (PS error grows ~1/albedo). Measured 2026-10-01 (synthetic board at the planned pixel density, a guessed 14-bit sensor): the verdict copes unaided down to albedo 0.008 (floor 0.23° at s = 5 mm), but the per-pixel error that the baked normal map carries passes 1° below about albedo 0.06 (1.43° at 0.03, 3.24° at 0.008); brackets at 4× and 16× bring those to 0.33° and 0.65°. With a noisier sensor (0.5 % read noise) albedo 0.03 is INCONCLUSIVE at one exposure (floor 3.57°) and fine bracketed (0.041°). Black, navy and dark batik: shoot with `ps_brackets`, and rerun `bench_dark_fabric.py` with the pilot's `radiometry.json` for the real numbers;
  - sheen (satin, silk, sateen, polyester, coated nylon; needs cross-polarisation);
  - pile (velvet, velour, corduroy; not Lambertian);
  - brushed or fuzzy fleece, and suede or leather;
  - glossy or metallic inks, and fluorescent dyes (colour);
  - thin parts (cords, straps, fringe), and deep folds (interreflection; concavity flag);
  - garments longer than the ~0.9 m framing.
- **Not supported:**
  - sheer or translucent fabric (chiffon, organza, tulle, mesh, lace);
  - metallic or mirror-like surfaces (sequins, lamé, foil, metallic-thread songket, patent leather, PVC), since cross-polarisation removes nearly all of their signal;
  - retroreflective trims and long-pile faux fur;
  - anything that moves during a ~45 min capture.
- **Per new fabric:** shoot a swatch on the ChArUco board (stage 6) before the garment. A fabric floor above 3° at s ≥ 5 mm means INCONCLUSIVE: decline, or warn.
- **A scan is not a personal try-on.** It is a static drape on one mannequin. In Coresight's 2023 survey of apparel sellers, 53 % named size and fit as the top return reason, and 16 % named colour. Fit needs pattern-based simulation or sizing tools. The scan's strengths are measured colour and true scale.

## Conventions: this file is the single source of truth
- **Camera frame:** OpenCV/COLMAP, with x right, y down, z forward. `X_cam = R X_world + t`, where R comes from the `images.txt` quaternion (world → camera).
- **Pixel centres:** COLMAP convention. The centre of pixel (row i, col j) is at (u, v) = (j + 0.5, i + 0.5), and the ray through it is ∝ ((u − cx)/fx, (v − cy)/fy, 1). `pycolmap.Camera.img_from_cam` is the reference implementation.
- **Scale s:** the standard deviation σ of a Gaussian on the surface, in mm. In a view, σ_px = fx · (s / 1000) / Z_med, where Z_med is the median garment depth in metres.
  - A feature of wavelength λ keeps at least 50 % of its amplitude when λ ≳ 5.3 s.
  - *Fine* means s ≤ 2 mm: weave, fuzz, stitching.
  - *Mid/low* means s ≥ 5 mm, i.e. wavelengths of roughly 27 mm and up: folds, seams, panel bulges.
- **Normals:** unit length and outward-pointing, so visible surfaces satisfy n · (C − X) > 0. Compare normals in the camera frame.
- **Light model:** I = ρ · E · max(0, −l·a)^μ · max(0, n·l) / r². Here:
  - l is the unit vector from the surface point to the light, and r is the distance;
  - a is the LED's axis, pointing into the scene;
  - μ is the LED's falloff exponent, with μ = 0 for an isotropic source.
  - Model a softbox or other area light as a grid of emitter elements. Solving a 60 cm softbox at 1.5 m as a point light gives about 1° median error (evidence B).
- **Light calibration:** stored per physical camera.
  - If the lights are fixed in the studio (the assumption), store them in that camera's frame. The values are then **constant across turntable steps; don't apply a per-step rotation.**
  - If the lights ride on the turntable, store them in the world frame.
  - The world-frame-plus-turntable-rotation scheme in `drafts/ps_disagreement_check.py` gives the same answer only if the turntable sign, step angle, step numbering and COLMAP pose are all exact. Use it only as a cross-check.
- **Radiometry:** PS needs linear intensities. With RAW, still verify linearity on the actual camera (exposure sweep and additivity; see Rig → camera).
  - Prefer RAW via rawpy: linear output, no auto-brightening, 16-bit, `user_flip=0`.
  - Without RAW, calibrate the camera response from bracketed exposures (`cv2.createCalibrateDebevec`) and pass the linearity gate in stage C. Camera JPEGs use vendor tone curves, not plain sRGB, so don't just invert sRGB.
  - Always subtract the ambient frame.
- **Undistortion:** GW accepts only PINHOLE/SIMPLE_PINHOLE cameras. Everything downstream uses the undistorted model from `colmap image_undistorter`. That command only processes registered images, so undistort every lit PS image yourself with the same mapping: from the original OPENCV camera to the undistorted PINHOLE K and image size, e.g. with `cv2.initUndistortRectifyMap` + `remap`.
  - **As built:** `undistort_lit.py --manifest <any manifest> --distorted <COLMAP cameras.txt | intrinsics JSON> --undistorted <COLMAP undistorted cameras.txt | auto>`.
    - Accepted distorted models: SIMPLE_RADIAL, RADIAL, OPENCV and FULL_OPENCV (COLMAP self-calibration defaults to SIMPLE_RADIAL). `auto` picks the PINHOLE camera `image_undistorter` would, via pycolmap.
    - Interpolation is bilinear, as COLMAP does; its positive weights keep intensities linear.
    - Any output pixel that draws on a saturated source pixel is written as full scale, so it stays flagged.
    - Measured against `pycolmap.undistort_image`: identical target camera, mean difference 0.2 grey levels.
  - **Half-pixel trap:** OpenCV's maps assume integer pixel centres, so both principal points are shifted by −0.5 first. When the undistorted camera keeps the focal length, as COLMAP's does, the two shifts cancel and a naive call looks right. With a different focal (0.7× in the test), the naive map is 0.15 px off, while ours is within 0.01 px.
  - `cv2.remap` quantises sampling positions to 1/32 px, identically for every light.
- **Units:** metres in data. Report surface scales in mm and angles in degrees.

## Data contracts
- **`config/lights.json`:** `{"cameras": [{"camera_id": 1, "frame": "camera"|"world", "lights": [{"id": 1, "role": "ps"|"reference", "kind": "point"|"area", "position_m": [x, y, z], "E": 1.0, "axis": [ax, ay, az], "mu": 0.0, "size_m": [w, h]}]}]}`. Only `role: "ps"` lights enter the solve.
- **`config/manifest.csv`:** columns `step, camera_id, colmap_image_name, light_id, path`, with `light_id` = `ambient` for the dark frame. Every stage reads images through this manifest; no other code parses filenames.
  - **Exposure brackets (2026-10-01):** an optional `exposure_s` column, then set on every row. A view's shortest exposure (the PS shutter) is its ordinary `lights`/`ambient`, so code that knows nothing of brackets sees an ordinary view; the longer ones are `Manifest.views[...]["brackets"]` and must repeat the base's lights and ambient. Undistortion and stage 1 keep the column (they copy every column).
- **`config/calib_manifest.csv`** (calibration and control captures): columns `target, position, camera_id, light_id, path`.
  - `target`: `intrinsics`, `mirror_ball`, `matte_ball` or `fabric_board`.
  - `position`: labels one placement. The mirror and matte ball on the same seat share a label.
  - `light_id`: a PS light id, `ambient`, `silhouette` (the backlit matte-ball frame) or `all` (any well-lit frame; for the board, the ambient-subtracted sum of the PS frames is used if `all` is missing).
- **`config/board.json`:** `{"dictionary": "DICT_5X5_250", "squares_x": 11, "squares_y": 9, "square_length_m": 0.03, "marker_length_m": 0.022, "legacy_pattern": false, "fabric_region_m": [x0, y0, x1, y1], "fabric_thickness_m": 0.001}`. Board frame = OpenCV's: origin at the printed top-left outer corner, x along squares_x, y down the print, z into the board, so the printed face points along −z. Corners within half a square of the fabric are ignored. `legacy_pattern` must match the generator the print came from (OpenCV < 4.6 boards use the legacy layout).
- **Intrinsics JSON** (`stageC_intrinsics.py`): COLMAP OPENCV model `params = [fx, fy, cx, cy, k1, k2, p1, p2]` in COLMAP pixel convention, plus RMS, per-view residuals and corner coverage. Only calibration code accepts it; everything downstream stays PINHOLE-only.
- **`config/calib_manifest.csv` additions (2026-09-27):**
  - Targets `dark`, `flat`, `sweep`, `drift` and `colour_chart`.
  - An optional `exposure_s` column, set for sweep frames.
  - Light combinations as light ids such as `1+2`, for the additivity check.
  - Parallel-polariser board placements are named `<placement>_par`; stage 6 turns them into a separate `fabric_parallel` floor.
- **`radiometry.json`** (`stageC_radiometry.py`): `{"noise": {"read", "full_well"}, "noise_fit": {...}, "checks": {"linearity", "additivity", "drift"}, "pass", "meta"}`. Units are fractions of full scale; `full_well` is in units per full scale.
- **`config/rig.json`** (see `config/rig.example.json`):
  - camera: the gphoto2 settings to lock (brand-specific names), the file extension, `shutter_setting` (the settings key holding the PS shutter) and `all_lights_shutter` (a plan may override either);
  - PS light ids and extra channels (such as the backdrop);
  - the light driver (serial line protocol or manual; `verify`: read every reply and check `?` → `STATE …`, default true, see `firmware/pico2_lights/README.md`) and the turntable driver;
  - the pilot and garment plans.
- **`chart.json`:** the ColorChecker's placement in the board frame: `{"origin_mm", "pitch_mm", "sample_mm", "rows", "cols"}` (first patch centre, patch pitch, sampled square). **`colour_correction.json`** holds `{"method", "matrix", "fits", "measured_camera_rgb", "reference_linear_srgb"}`.
- **Masks** (`stage_masks.py`): `<session>/masks/<colmap_image_name>`, 8-bit PNG in the geometry of `colmap_images/`, 255 = object (garment, mannequin, turntable top), plus `masks.json`. The static exclusion is an 8-bit PNG the size of the images, white = never object.
- **`stage_sfm.py` output:** `<out>/colmap/` (database, raw models), `<out>/aligned/` (text, distorted camera, turntable frame), `<out>/undistorted/` (`images/`, `sparse/` as text PINHOLE, `masks/`), `<out>/lit/` (every manifest frame undistorted, `manifest.csv`), `sfm.json`, `sfm.md`; exit code 1 if a gate fails (all images registered; reprojection ≤ 1 px; camera centres within 3 mm RMS of a circle; board frames within 1 mm RMS of one axis; SfM and board axes within 0.5°; tape distance within 1 % if given).
- **Stage 1 output:** a GW dataset `images/` (RGBA with alpha = mask when `--masks` is given; the images must be PNG), `sparse/0/` (with POINTS2D shifted if cropped), `lit/` + `manifest.csv`, and `stage1.json` (gate, crop, masks, and the GW command). **Stage 2:** `stage2.json`, `stage2.md` and `overlays/` (mesh silhouette red, mask green; `mesh_hits_outside_mask` per view); exit code 1 on failure.
- **Albedo:** `<out>/<stem>.npz` with `albedo_rgb` (linear sRGB, absolute) and `valid`, plus `<stem>_albedo_srgb.png`. **Bake:** `<out>.glb` (glTF 2.0, +Y up, explicit TANGENT, baseColor sRGB, normalTexture +X right / +Y up, metallic 0, optional `KHR_materials_sheen`), plus `<out>.json`, `_normal.png` and `_basecolor.png`.
- **Delivery limits for the `.glb`** (checked 2026-09-27):
  - **Android Scene Viewer:** GLB ≤ 15 MB (10 MB recommended), textures ≤ 2048², ≤ 100k triangles (30–50k ideal). Its only extensions are KHR_materials_unlit and KHR_texture_transform, so sheen is ignored.
  - **iOS:** AR Quick Look needs USDZ, which model-viewer generates in Safari and Shopify converts to. USD preview materials have no sheen. Safari 27's `<model>` element also takes GLB.
  - **HarmonyOS:** ArkGraphics 3D loads `.glb`/`.gltf`.
  - **Size:** the synthetic bake (1024² textures) is 4.3 MB; at the default 2048² it measured **9.06 MB** with PNG textures (2026-10-01; the earlier estimate was ~12 MB). A real weave may compress worse. **Built (2026-10-01):** `stage_bake.py --max-mb` (default 15, Scene Viewer's limit; 10 for its recommendation) stores the base colour as JPEG at falling quality (95…75) when the GLB with PNGs is bigger, and keeps the normal map PNG, whose JPEG blocks would read as tilted normals. Exit code 1 if even q75 does not fit. Measured on the synthetic textures: a budget of 60 % of the PNG size took JPEG q95 (2.41 → 1.13 MB), base colour ΔE00 against the PNG median 0.74; the normal map byte-identical. Test: `tests/test_N_bake.py`.
- **Undistorted images** (`undistort_lit.py --out D`): `D/<same relative path>.png`, 16-bit linear grey (the channel mean), saturated = 65535; `D/<manifest name>` with only the `path` column changed; `D/cameras.txt` with the PINHOLE camera(s); `D/meta.json`.
- **Stage C report** (`stageC_lights.py --report-dir`): `stageC_lights.md/json` with ball centres, per-light triangulation residuals, E / axis / μ with uncertainties, leave-one-position-out prediction errors, and both light-position estimates (mirror and shading) with their distance ("mirror vs shading").
- **Per-view outputs:** `runs/<name>/<stage>/<colmap_image_stem>.npz`, float32 arrays in the camera frame, plus a `meta.json` per stage with inputs, parameters and git hash.
  - Heatmaps are PLY files with one per-face scalar per scale.
  - Summaries are JSON plus a short markdown file.
- **Garment selection:** `config/garment_faces.npy`, a boolean per face of the raw GW mesh, or a bounding volume in the world frame. Apply it through the face-id map. The mannequin and turntable are excluded from every statistic.
  - **As built (`stage_garment.py`, 2026-09-28):** `--mesh` (turntable frame) → `<out>.npy`, `<out>.ply` (garment red, the rest blue) and `<out>.json` (per-criterion counts and areas; registration R, t, inlier fractions, runner-up). `--reference`: the bare-mannequin raw GW mesh in its own turntable frame. `--garment-masks`: 8-bit PNGs named like the views (255 = garment) in the geometry of stage 3's views, with `--stage3` for the face-id maps.
- **Stage 5 `faces.npz` additions (2026-09-28):** `n_bins_<s>`, `dmean_<s>` (mean resultant vector, world frame, float32) and `consistency_bin_deg`. **Stage 7:** clusters carry `vertical_fraction`, `views_median`, `bins_median`; the verdict JSON may carry `vertical_warning` {clusters, of, all}.
- **2026-10-01 additions.** `faces.npz`: `elev` (each face's mean Y/Z in the camera over its views). `floors.json`: `<kind>_placements` = [{`name`, `elev` [lo, hi], `floor_deg` {s: p95}}] per control view. Verdict JSON: `row_coverage` {`covered_area_fraction`, `pass`, `garment_elev`, `uncovered_elev`, `placements`} (None with older stage 5/6 output); clusters `elev`, `elev_covered_fraction`, `height_placements`, `floor_at_height_deg`, `exceeds_floor_at_height`, and with stage 5's inputs `height_mm` / `height_fine_mm` {`peak_mm`, `median_mm`, `views`, `per_view_peak_mm`}; `vertical_warning` gains `uncovered` and `within_height_floor`. Stage 4 `meta.json`: per view `exposures_s` and `bracket_fraction` when bracketed. Bake `<out>.json`: `glb` {`bytes`, `max_bytes`, `base_colour` (`png` or `jpeg q<N>`), `within`}.

## Verified facts about Gaussian Wrapping (repo @ 11e3b6f, 2026-08-28)
- **Run command:** `python gaussian_wrapping/scripts/train_and_extract_gw_radegs.py -s <colmap_dataset> -m <out> --no_postprocess [--isosurface_value 0.2] [-r N]`
- **Output mesh (radegs):** the raw mesh `<out>/mesh_exact_computation_2pivots[_transmittance_threshold_<0.5+iso>]_searched.ply` is **always** written first. Post-processing, on by default, then writes `…_post.ply`, which keeps only the largest connected component. Use the raw mesh; earlier runs can be reused. The `ours` variant writes `mesh_ours_2pivots[_post].ply`.
- **Texture refinement** writes `<mesh stem>_texture_refined_<n_iter-1>.ply` (`_999` by default). Only vertex colours are optimised; the geometry is identical.
- **Image downscaling:** images wider than 1600 px are downscaled to 1600 px unless `-r` is given. `-r 1` keeps full resolution. Values other than 1, 2, 4 or 8 set the target width. Record which `-r` was used.
- **GW ignores the principal point.** `scene/cameras.py` hardcodes Cx = (W−1)/2 and Cy = (H−1)/2, and `scene/dataset_readers.py` keeps only the field of view.
  - COLMAP self-calibration keeps the principal point centred, because it isn't refined by default, and undistortion preserves that: tested in pycolmap 4.2, a centred input stays at +0.00 px.
  - An externally calibrated off-centre point stays off-centre after undistortion: +40 px in gives +41.9 px out.
  - In that case GW is silently wrong. Hence the gate in stage 1.
- **Oriented-normal alignment** lives in `regularization/regularizer/normal_field.py::compute_normal_field_regularization`. It renders per-Gaussian oriented normals in the world frame and aligns them with normals from the rendered median depth.
- **Default schedule** (`configs/normal_field/default_regular_densification.yaml`):
  - the normal field starts at iteration 20,001;
  - flip-clone densification runs from 22k to 26k, every 1k iterations, on the top 5 % of normal errors;
  - training lasts 30k iterations, and extraction hardcodes `--iteration 30000`.
- **`--depth_order`** computes DepthAnythingV2 priors at start-up. Its weight is 1 from iteration 3k, 0.1 from 7k, 0.01 from 15k, and it switches off at 15k. It's a code pattern to copy, not a normal hook.
- **Hardware:** GW needs CUDA, so run it on the NVIDIA machine. `install.py` pins torch 2.3.1 with CUDA 11.8 or 12.1. An RTX 50-series (Blackwell) card needs a newer torch/CUDA, so expect to adapt the install. Keep GW in its own conda env (Python 3.9). The diagnostic runs on CPU in a separate env.

## Pipeline: stages and gates
Build one small CLI script per stage under `src/`. No stage reads another stage's internals.

0. **Environment.** Python ≥ 3.10 with numpy, scipy, trimesh, embreex, opencv-python-headless, rawpy, pycolmap and pillow. Construct `trimesh.ray.ray_pyembree.RayMeshIntersector` explicitly (see Gotchas).

C. **Calibration.** Run this once per rig change, before any PS solve on real data.
   - **Metric scale** (built 2026-09-28, `stage_sfm.py`): the garment session's metric-board frames (board upright on the turntable at −40…40°) are fitted with one rotation axis in pixels; the camera's distance from that axis in metres over the SfM circle radius is the scale. Checks: board frames within 1 mm RMS of one axis, SfM and board axes within 0.5°, and a tape-measured camera-to-axis distance within 1 % if given (coarse: the camera centre is the entrance pupil). Synthetic: 1.5998 m against 1.6000 m.
   - **Light positions:** place a mirror sphere at ≥ 3 positions spanning the garment volume and triangulate each light from the reflected highlight rays. Target ≤ 5–10 mm: an error of 10 mm gives about 0.5° median, 20 mm about 1.3° (evidence B).
     - **As built:** a mirror ball and a matte (primer) ball of the same radius share one seat and are swapped at each position. A mirror ball's limb reflects its background, so its own silhouette is unusable; the centre comes from a **backlit** frame of the matte ball (dark ball on a bright, even backdrop). The centre is the cone fit of sub-pixel edge points, with the stand rejected as outliers. Synthetic: 1–3 µm lateral, < 0.1 mm in depth.
     - **Seat repeatability dominates.** A lateral centre error dC tilts every reflected ray by ~2 dC/R, so the light moves ~2 L dC/R: about 6 mm per 0.1 mm with R = 40 mm and L = 1.25 m. Synthetic, 6 positions: exact seat 1.2 / 3.8 mm (median / max); 20 µm (1σ per axis) 6.8 / 10.7 mm; 100 µm 11.3 / 20.9 mm. So the mirror estimate needs a kinematic seat repeatable to ≲ 10 µm, and radii measured to ~0.01 mm. Primer thickness changes where a ball sits in a ring seat.
     - **Second estimate:** refit each light's position from the matte ball's shading. It is independent of the seat and of the mirror ball, and on the synthetic (Lambertian) ball it gets 0.6–2 mm with 6 positions (2.8 / 8.4 mm with 3). It assumes Lambertian primer, which is unverified. The report gives "mirror vs shading" per light, which tracked the true mirror error to within ~1 mm on synthetic data. `lights.json` uses the mirror estimate unless `--position-source shading`. **Decide which to trust from the pilot's agreement.**
     - Use **≥ 6 positions** (3 heights × 2 lateral, with ±8 cm of depth spread) and a ball of radius ≥ 40 mm; 3 positions roughly doubles the errors.
   - **Intensity and LED falloff:** fit E (within 2 %) and, where relevant, the LED axis and μ, from a matte white sphere or a white board at several positions.
     - **As built:** a matte-ball fit of E·ρ_primer, axis and μ per light, with light positions fixed (PS albedo therefore comes out relative to the primer). Judge it by leave-one-position-out prediction, not by μ and axis individually. Synthetic: relative E within 0.65 %, irradiance over the garment volume within 1.2 %, leave-one-out within 1.7 %.
     - Fitting E after the positions absorbs much of the position error: with the mirror positions (errors of several mm), the calibrated fabric floor was within ~0.05° of the true-light floor at s ≥ 5 mm.
   - **Intrinsics for the pilot:** ~15 ChArUco shots at varied poses covering the frame (`stageC_intrinsics.py`, OPENCV model). The principal point is only weakly determined (synthetic: 1.5 px off, equivalent to a 0.03° global rotation of the camera frame, which is harmless because lights, poses and normals all share that frame). The residual ray error after that rotation is ≤ 0.3 px within the corner coverage. **After the pilot, don't move, refocus or zoom the camera:** the lights are stored in its frame.
   - **Camera response** (only without RAW): gate on linearity, with a linear-fit R² > 0.999 over the intensity range actually used.

1. **GW reconstruction** (on the RTX server).
   - **Gate first:** in the undistorted model, |cx − W/2| < 0.5 px and |cy − H/2| < 0.5 px. If it fails, crop the undistorted and the lit images identically so the principal point is centred, and update `cameras.txt`.
   - Run the radegs script with `--no_postprocess` and keep the raw mesh.
   - With `--masks`, stage 1 writes the object mask as the images' alpha: GW composites it onto its (black) background in training (`train.py` 291–293) and, with `--use_valid_mask` (on in the radegs script's extraction flags), treats pivots outside every view's mask as empty, so the mask must cover the garment, the mannequin and the turntable top.

2. **Frame gate.** Stop if any check fails.
   - **(a) Reprojection:** project the sparse points observed in each image with *your* projection code. The median distance to COLMAP's POINTS2D observations must be within about 0.2 px of COLMAP's mean reprojection error, **and** the mean signed residual must be under 0.1 px on each axis. The second condition catches half-pixel shifts.
   - **(b) Depth:** compare ray-cast mesh depth with sparse-point depth at the same pixels. The median |Δ| must be a few mm, not cm.
   - **(c) Lit-image alignment:** each lit image must line up with that step's SfM image to within 0.2 px (e.g. ECC or phase correlation on a textured patch). This catches RAW crop, size and orientation mismatches. Exposure brackets (dark garments, 2026-10-01) are checked against the base frames scaled by the exposure ratio and clipped where the bracket clips: the set within 0.2 px, each frame within 0.3 px of its own light's base frame (`gates.bracket_alignment`; clean frames 0.03–0.11 px, a 1 px bump 0.91 px, synthetic).
   - **(d) Overlay:** write PNGs of the mesh silhouette drawn on the photos, for a human to glance at.

3. **Mesh maps per view** by embree ray casting in COLMAP convention.
   - Produce the hit mask, depth, 3-D position, face id and face normal, all in the camera frame.
   - Report the fraction of back-facing hits. Above 1 % means inconsistent winding; investigate rather than silently flip normals.
   - Batch rays to stay within memory. This stage replaces the Blender renderer.

4. **PS solve per view** with the near-light model above.
   - Compute each pixel's light vectors and distances from stage 3's positions and the calibration. Mesh position errors of a few mm cost about 0.3° (evidence B).
   - For each pixel, reject individual lights that are shadowed or saturated. Solve when ≥ 4 lights remain; allow 3 but flag the pixel as low confidence.
   - This is a batched 3×3 normal-equation solve per pixel. One pseudo-inverse per shadow pattern no longer works, because every pixel has its own light matrix.
   - Output: n_ps in the camera frame, albedo, residual-based confidence, and the number of lights used.

5. **Compare** (per view, in the camera frame, garment faces only).
   - **Valid pixels** must satisfy all of: a garment mesh hit; a valid PS solve; angle between n_mesh and the direction to the camera below 75° (validated earlier against synthetic ground truth); at least 3 px away from any depth discontinuity; confidence above the minimum.
   - **Per-view rotation:** fit a weighted Kabsch rotation from n_ps to n_mesh and report its angle. Remove it before the steps below. A rotation that stays consistent across views means light calibration or frame error.
   - **Vector low-pass at each scale s ∈ {1, 2, 5, 10, 20, 50} mm:**
     - apply a masked Gaussian (weights = valid × confidence) to each normal component, then renormalise;
     - θ_s = ∠(LP_s n_ps, LP_s n_mesh);
     - d_s = Rᵀ(LP_s n_ps − LP_s n_mesh), the disagreement vector in the world frame.
     - **Never low-pass the angle map itself** (evidence A).
   - **Per face,** accumulate with `np.add.at` over views:
     - θ̄_s(f), the multi-view mean of θ_s;
     - the number of views that see the face;
     - the mean resultant length of the d_s directions (the world-frame consistency).
     - A real shape error is fixed on the garment, so it points the same way in every view. PS bias rotates with the lights relative to the garment, so it doesn't.
   - **Concavity flag** per face, from the mesh's mean curvature at scale s. Interreflections live in concavities, and neither control can measure them.
   - **Outputs:** one heatmap PLY per scale, plus summary statistics per view and pooled.
   - **Optional, built 2026-10-01 (in stage 7):** integrate the low-passed normal difference over a region to get an implied height error in mm (`gwps/height.py`; see step 7).

6. **Control captures** on the same rig with the same settings. Run stages 3–5 with the known geometry in place of the GW mesh.
   - **(a) Matte sphere of known radius** (e.g. a ball painted with primer), at ≥ 3 heights spanning the garment volume. Get its pose from its silhouette, the known radius and K, and cross-check against the mirror-sphere positions.
   - **(b) The garment fabric on a rigid board framed by ChArUco markers,** at tilts of 0°, ±30° and ±50°. Get the board's pose with `solvePnP`. Measure PS against the **ChArUco plane normal**, not the mean PS normal, which would hide a uniform sheen bias.
   - **Floor:** floor_s = the 95th percentile of θ_s over valid control pixels, per scale, with the fabric floor and the sphere floor kept separate.
   - **As built** (`stage6_controls.py`):
     - For the board, world = board frame, one "image" per placement with the ChArUco pose, and the fabric rectangle as the mesh.
     - For the ball, world = ball frame, pose t = centre, and an icosphere with 7 subdivisions as the mesh.
     - Stages 3–5 then run unchanged, without per-view rotation removal.
     - The sphere floor is optimistic, because the same matte-ball images fit E / axis / μ; the verdict uses the fabric floor.
     - Synthetic pilot (noise as in the torso tests, 20 µm seat): fabric floor 0.094 / 0.071 / 0.063 / 0.058° at s = 5 / 10 / 20 / 50 mm with calibrated lights and estimated poses, against 0.071 / 0.039 / 0.023 / 0.014° with the true ones.
   - **Board placements:** also capture it at ≥ 3 heights covering the garment's image rows (vertical-bias blind spot, see Decisions). Stage 7 checks the coverage (`row_coverage`, 2026-10-01).
   - **Never exactly square-on.** A square-on board's normal rests on weak perspective: 0.1° of tilt moves the corners of a 33 cm board at 1.45 m by only ~0.07 px. Noise-free synthetic renders gave normal errors of 0.07–0.17° at 0–15° tilt about the horizontal axis, from sub-0.05 px corner errors alone, against a median 0.019° and max 0.054° at 20°. So the "0°" placements are tilted 20° about the horizontal axis, and the ±30° / ±50° ones stay as they are.
   - **ChArUco board design:** `bikin_papan_charuco.py`, given the camera (intrinsics JSON, cameras.txt, or fx / focal and pixel pitch) and the distance, chooses the board:
     - squares of ~60 px at the garment distance, but ≥ 15 mm and ≤ 250 markers;
     - 5×5 ArUco cells ≥ 4 px even at 50° tilt;
     - a blank fabric window framed by 2 squares.
     - It writes an exact vector PDF (verified: scale error < 0.01 %, corners exact at 508 dpi), a PNG preview, `board.json`, and a self-check that renders the board through the camera and detects it.

7. **Verdict.** Work from per-face θ̄_s, over garment faces seen in ≥ 2 views. A *cluster* at scale s is a connected set of faces with θ̄_s > fabric floor_s + 1° and total area ≥ (2s)². Apply these rules in order:
   1. **INCONCLUSIVE** if the fabric floor is above 3° at any s ≥ 5 mm.
   2. **BAKE** if no cluster exists at any s ≥ 5 mm.
   3. **BUILD candidate** if at least one cluster at s ≥ 5 mm has a mean resultant length of d_s ≥ 0.7 across views. Report its concavity and silhouette flags alongside.
   4. **UNEXPLAINED** otherwise: clusters exist, but they rotate with the lights, so they look like PS bias. Report them and don't build.

   Before acting on a BUILD candidate, rerun GW at `-r 1`, because the default 1600 px training width can be the cause. All margins are starting points; revisit them after the controls. Write the verdict and supporting numbers to `runs/<name>/verdict.md`.

   **Also reported (2026-10-01), without changing the rules:** board coverage of the garment's elevations (the blind spot, see Decisions); and, given stage 5's inputs (`--sparse --stage3 --stage4`), each cluster's **implied height** in mm: n_mesh − n_ps, low-passed as in stage 5, integrated over the cluster with its holes filled, against a ring 2σ around it, by masked least squares in each view within 45° of face-on, median over those views. `height_mm` is at the cluster's own scale and `height_fine_mm` at 2 mm; > 0 = the surface PS sees lies outside the mesh (GW low). The missing 3 mm bump (σ 10 mm, noisy): +2.15 mm at s = 5 mm, +1.32 mm at 10 mm, +2.62 mm at 2 mm (s-low-passed apex 2.4 / 1.5 / 2.9 mm; the ring still sits on the bump's tail). Tests: `tests/test_I_height.py`.

## Synthetic tests: all must pass before touching real data (`tests/`, pytest)
Build a synthetic turntable scene with known poses and mesh, rendered with the forward model above. Use two noise settings: none, and additive Gaussian noise of 0.5 % of the maximum linear intensity plus Poisson shot noise. Wherever a test uses the verdict rule, take floor_s from a synthetic flat-board control rendered with the same noise setting. Then check that:
1. **PS accuracy:** median normal error < 0.1° without noise and < 1° with noise, in unshadowed, non-grazing pixels.
2. **No false alarms:** when the mesh equals the ground truth, θ̄_s < 0.3° without noise, and there are no clusters with noise.
3. **A real bump is found:** remove a smooth bump (Gaussian with σ = 10 mm, height 3 mm) from the mesh. The verdict rule must flag a cluster at s = 5 mm and s = 10 mm that overlaps the bump, with mean resultant length ≥ 0.7.
4. **Fine texture is ignored:** PS normals that differ from the mesh only by a ripple with a 3 px period and ±10° amplitude give no cluster and θ̄_s < 0.5° for every s ≥ 5 mm. This guards against evidence A.
5. **Bias rotating with the lights is rejected:** add a smooth bias defined in the camera frame, so it rotates with the lights across steps. It must fail the consistency check (UNEXPLAINED), while test 3's bump passes it.
6. **Near-light matters:** a directional-light solve on near-light data reproduces evidence B, and the near-light solve removes the bias.
7. **Parsing:** an `images.txt` with an empty POINTS2D line parses correctly.
8. **Projection:** your projection matches `pycolmap.Camera.img_from_cam` to within 0.01 px for a PINHOLE camera with an off-centre principal point. Rays cast through projected vertex pixels hit within 0.1 mm of the visible vertex.

Tests 3 and 5 run at 12 × 30° and at the garment plan's 36 × 10° (2026-09-28). Beyond the eight: `test_V_*` (consistency bins, vertical share), `test_M_*` (masks), `test_S_*` (the COLMAP chain on a synthetic garment session, through stage 2 with the true mesh), `test_Y_*` (garment selection), `test_Z_*` (2M faces through stages 3–7 and the bake), `test_W_*` (the GW dry-run kit and the whole chain after capture with PS), `test_L_*` (the light-controller firmware and `SerialLights` verification), `test_D_*` (dark fabrics: exposure brackets and their merge), `test_H_*` (board coverage of the garment's rows), `test_I_*` (implied height of a cluster), and the stage C, capture, gate, albedo and bake tests listed under Status.

## Known bugs in drafts/ (verified 2026-09-25; repro in docs/evidence/)
- **`render_mesh_normals.py`**
  - It crashes on Blender 5.0: `scene.node_tree` was removed.
  - The `shift_x` sign is wrong. A principal point 80 px right of centre puts the render 160 px off (IoU 0.00); with the sign fixed, IoU is 0.998.
  - The Cycles Normal pass is in **world** space: 0.6° error against world vs 85° against the OpenCV camera frame. The docstring says camera space.
  - Replace it with the stage 3 ray-caster.
- **`ps_disagreement_check.py`**
  - It compares world-space mesh normals with camera-space PS normals.
  - It low-passes the angle map. Weave-only differences of ±6° / ±12° then read as 2.9° / 5.8° of fake low-frequency disagreement; vector low-pass gives 0.1° / 0.2°.
  - Its directional-light model gives about 9° error 10 cm from the centre with lights 1.5 m away, and about 21° median over the garment.
  - It treats JPEG values as linear, which costs about 11° median.
  - It doesn't subtract ambient light.
  - It feeds every light in the file into the solve, including the non-PS `diffuse_reference`, and pairs images to lights by sort order instead of by id. With only the 3 PS lights, its residual term would be identically 0.
  - It drops any pixel shadowed under any light, so folds are under-sampled.
  - It has no grazing-angle cutoff.
  - It rotates world-frame lights per turntable step, which is sensitive to sign, step angle and pose.
- **`lights.example.json`**
  - It says to measure "from the light toward the capture volume centre". PS needs surface → light, so following it literally flips every normal.
  - It stores directions only, but the near-light model needs positions.
- **`build_camera_list.py`**
  - It drops blank POINTS2D lines and silently misparses the rest. In the test, the 3rd image is lost and a bogus image "70" appears.
  - It treats OPENCV, fisheye and RADIAL models as pinhole. Reject anything that is not PINHOLE/SIMPLE_PINHOLE.
- **`ps_disagreement_diagnostic.py`**
  - In pycolmap 4.x, `image.cam_from_world` is a method; call it.
  - Each stored sample `normals_world[y, x]` is a numpy *view*, so it keeps that view's whole normal array alive: about 576 MB per 24 MP view, on top of roughly 235 B per sample. Memory runs out after a few views. Accumulate sums with `np.add.at` instead.
  - Its "resolvable" band is the single-face scale, which is the noisiest scale, not a low/mid-frequency measure.
  - It casts rays through (j, i) instead of (j + 0.5, i + 0.5).
  - It shares the PS model problems listed above.
- **`run_diagnostic.sh`**
  - It runs the `ours` script with `-r 2` instead of radegs.
  - Its guessed mesh filenames are wrong (see Verified facts).
  - It doesn't pass `--no_postprocess`.

## Gotchas
- **Ray casting.** trimesh's pure-Python `ray_triangle` fallback runs out of memory: under 10k rays against a 5k-face mesh exhausts a 3.9 GB container. trimesh 5.x uses embree automatically when embreex is installed, but construct `RayMeshIntersector` explicitly so a missing embreex fails loudly. Measured: 1 M rays against 327k faces in ≈ 0.35 s and 460 MB.
- **COLMAP `images.txt`** has two lines per image, and the second may be empty. Don't filter out blank lines.
- **Decimation** happens after the diagnostic, only for the texturing path: `stage_bake.py`, quadric decimation (trimesh → fast_simplification) to 50k faces by default.
- **ChArUco pixel convention (OpenCV 4.11).** ArUco marker corners and `cornerSubPix` put pixel centres at integer coordinates, but `CharucoDetector.detectBoard` returns ChArUco corners **+0.5 px** from that (measured +0.49 / +0.47 px against truth), i.e. at COLMAP's convention by accident. `gwps.charuco.detect` re-refines every corner with `cornerSubPix`, whose convention is known, then adds exactly +0.5. This is also more precise (RMS 0.084 against 0.130 px). Don't feed raw `CharucoDetector` corners to code with a known convention.
- **ChArUco near the fabric window:** OpenCV also "detects" corners on the window's edge, where frame squares meet the blank window. `detect()` drops every corner within half a square of the window.
- **colour-science 0.4.7 pulls in numpy 2**, which breaks the pinned `bpy`; use colour-science 0.4.6 with numpy 1.26.4.
- **gphoto2:** Fedora 44 ships libgphoto2 2.5.33 (not installed yet). The Canon R6 Mark III is only in the development branch; the R5 Mark II and Nikon Z5 II are in neither.
- **COLMAP and a static background (2026-09-28).** With a textured static studio and COLMAP's defaults (no masks, `TwoViewGeometryOptions.filter_stationary_matches` off, as it is by default in pycolmap 4.2), incremental mapping registered **no model**. `stage_sfm` turns the stationary filter on and uses masks.
- **pycolmap 4.2 API:** `ImageReaderOptions.camera_params` is a comma-separated string; `Image.cam_from_world()` and `Image.projection_center()` are methods; `Reconstruction.transform(Sim3d(scale, Rotation3d(R), t))` maps new = s R old + t; `undistort_images` writes a binary `sparse/` (`stage_sfm` rewrites it as text); masks are `<mask_path>/<image name>.png` (`step0000.png.png`), black = no features.
- **COLMAP self-calibration fixes the principal point at the image centre** (not refined by default). With a 30 px offset and k2 / tangential distortion it put the scale 2.5 % off (synthetic, SIMPLE_RADIAL). Pass the pilot's intrinsics.
- **`/tmp` is a RAM-backed tmpfs with a ~6.3 GB per-user quota.** A full test run writes ~2 GB of synthetic images, so `pytest.ini` sets `--basetemp=runs/_pytest`, which pytest clears at the start of each run. When the quota filled, even the tool's own output capture failed.
- **A fresh clone has no `runs/`** (it is git-ignored), and pytest creates `--basetemp` but not its parent, so every test errored at setup (found 2026-10-01 in a cloud clone). `tests/conftest.py` now creates `runs/`. The same container also lacked GNU time (`/usr/bin/time`), which `gw_pod.sh run` and `bench_mesh_size.py` call, so `test_W_gw_dryrun.py::test_pod_script_run_and_pack` and `test_Z_mesh_size.py` failed on the unchanged code until `apt-get install time`.

## Before building L_PS (only after a BUILD verdict)
- Close the remaining gaps in the prior-art search (partly done 2026-10-01, see `docs/prior_art_report.md` → "Gap check": the code is still at 11e3b6f; web search found no citing work that supervises GW with PS normals; the issues, PRs and forks are still unread, and no citation index was reachable from the cloud session):
  - `gh issue list -R diego1401/GaussianWrapping --state all`, the same for PRs, and the repo's forks;
  - Google Scholar "cited by" for arXiv 2604.07337.
- **Hook:** add L_PS next to the median-depth alignment in `compute_normal_field_regularization`. Convert PS normals to world with the same `view_to_world_transform`, and resample them to GW's training resolution (area-average, then renormalise). Use GW's centred-principal-point camera, i.e. the cropped images from stage 1.
- **Schedule:** densification selects by a quantile of oriented-vs-median-depth error, so L_PS would change *which* Gaussians get cloned. Either start L_PS after 26k and extend training (the extraction iteration is hardcoded), or exclude PS-driven error from the selection.

## Working agreements
- Ask Dawud before large downloads, GPU runs longer than about 30 minutes, or deleting anything outside this directory.
- Update this file whenever a ⚠ item is answered or a decision changes.
- Give every reported number its units and the scale it refers to.
