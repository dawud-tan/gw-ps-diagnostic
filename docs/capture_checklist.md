# Capture checklist: pilot session (light calibration + fabric control)

No garment, no GW. Result: calibrated lights and the **fabric floor**; above 3° at s ≥ 5 mm means INCONCLUSIVE (fix PS before anything else). The reasons and numbers behind each item are in CLAUDE.md (Rig → camera, stage C, stage 6).
**Run it with `src/capture_session.py --config config/rig.json --plan pilot --out captures/<date>_pilot`**: it prompts each step below in this order, drives camera and lights, and writes `calib_manifest.csv` as it goes (`--mock` rehearses the order without hardware).

## Prepare
- [ ] **Lights fixed in the studio** (not on the turntable): 8 bare LEDs (6 minimum), ~1.2–1.5 m from the garment position, 30–45° off the camera axis, spread evenly in azimuth. **Wide beam** (the whole garment in the flat part of the beam; a ~15° spot covers only ~34 cm at 1.3 m), CRI > 95. **Constant-current (flicker-free) drivers.** Label each with its id.
- [ ] **Camera body:** confirm it in libgphoto2's supported list (remote capture and settings). RAW only; manual shutter, aperture, ISO, white balance and focus. **Turn off** in-body stabilisation, auto power-off, sensor cleaning at power-on/off, and in-camera lens corrections. Electronic shutter only if a flat frame shows no banding.
- [ ] **Lens and framing** (full frame, portrait): 35–50 mm at 1.0–1.5 m frames a 0.9 m garment (85 mm needs ~2.5 m). **f/11**; check that the nearest and farthest garment surfaces are both sharp. Tape the focus ring.
- [ ] **Cross-polarisation:** polarising gel on each light, linear polariser on the lens. With the chrome ball in place, rotate each light's gel until its highlight vanishes. Tune every light, not pairs. Mark the lens polariser's **crossed** and **parallel** positions. Expect ~3 stops less light.
- [ ] **Two balls, same diameter ≥ 80 mm:** one mirror (chrome), one primed matte white. Measure both diameters to 0.01 mm. Keep the mirror ball clean (gloves).
- [ ] **Kinematic seat** (3-point or ring) on a thin, dark, rigid stand. A swapped ball must return to within **~10 µm**: every 0.1 mm of seat error moves a light by ~6 mm.
- [ ] **Backdrop light** behind the ball positions (lit white sheet or LED panel), switchable on its own, for the **backlit silhouette** frames.
- [ ] **ChArUco board:** generate for your camera with `bikin_papan_charuco.py` (writes `board.json`). Print at 100 % on **matte** paper and glue it flat to a rigid backing (aluminium composite or glass). **Measure a 10-square span with callipers** and put the measured square size in `board.json`.
- [ ] **Fabric:** garment fabric taut and flat over the board's blank window, edges taped, no wrinkles. Note its thickness.
- [ ] **Colour and albedo references:** a ColorChecker, and a white or grey target of known reflectance (Spectralon or a measured grey card).
- [ ] **Warm up** the LEDs ≥ 15 min. Studio dark: nothing on except the light being captured.
- [ ] **Exposure:** in every single-light RAW, the matte ball and the fabric stay below ~90 % of full scale (mirror highlights may saturate). This is the **PS shutter** (`settings.shutterspeed` in `rig.json`); with the recommended LEDs at 1.3 m, f/11, it is ~0.4 s, or ~3 s cross-polarised. For a later dark garment, change exposure only in exact shutter steps and record the ratio, or bracket.
- [ ] **All-lights shutter** (`camera.all_lights_shutter` in `rig.json`): frames lit by more than two PS lights (flats, intrinsics, and later the SfM and metric-board frames) are 2.5–2.8 stops brighter and would clip at the PS shutter. Start ~3 stops (8×) shorter and check they also stay below ~90 %. The script switches between the two shutters by itself and reads each change back.

## Capture (per camera; do not touch the camera from here on; lens polariser **crossed** unless stated)
1. **Noise and linearity:** 10 dark frames (lens cap, PS shutter); 10 frames of an evenly lit matte white card (all-lights shutter); the same card at 6–8 shutter speeds (exposure sweep, bracketing the all-lights shutter, since the card is lit by all eight).
2. **Intrinsics:** ~15 board shots at 0.7–1.1× the garment distance, tilted up to ±35°, together reaching every edge and corner of the frame. Any lighting.
3. **Balls at ≥ 6 positions** spanning the garment volume (3 heights × left/right, ±8 cm in depth). At each position, without touching the seat:
   - matte ball: **silhouette** (backdrop on, PS lights off) · **ambient** (all off) · **each PS light alone**;
   - swap in the mirror ball, lens polariser **parallel**: **ambient** · **each PS light alone**; then back to crossed.
4. **Additivity:** matte ball at position 1 with **lights 1 and 2 on together** (light id `1+2`; compared with the sum of their single-light frames; a pair, not all eight, so nothing saturates).
5. **Fabric board**, each placement = **ambient + each PS light alone**, shot **crossed and again parallel** (the two fabric floors decide whether cross-polarisation is needed):
   - "face-on" at 3 heights (top, middle, bottom of where the garment appears), **tilted 20° about the horizontal axis**: a square-on board's normal is only good to ~0.1–0.15°;
   - ±30° and ±50° about the vertical axis, at mid height.
6. **Colour:** ColorChecker and the reflectance reference at the garment position, **each PS light alone** + ambient.
7. **Drift check:** repeat the matte ball at position 1 under light 1 at the very end. It must match the start within 1 %.
8. **Write down:** ball diameters, measured square size, fabric thickness, camera settings (shutter, aperture, ISO), polariser positions, light id ↔ physical light, file naming.

## After
- [ ] **Don't move, refocus or zoom the camera**: the lights are calibrated in its frame, and the garment session uses it. Don't power-cycle it *during* a session (a re-seated sensor misaligns the lit frames); between sessions it's harmless.
- [ ] Processing (the capture script wrote `calib_manifest.csv`):
  `stageC_radiometry` (noise model, linearity, additivity, drift) → `stageC_intrinsics` → `stageC_lights --noise radiometry.json` (check: mirror vs shading < 5 mm, leave-one-out < 2 %) → `undistort_lit` → `stage6_controls` (fabric floor crossed **and** parallel) → `stage_albedo calibrate` (colour correction from the chart).

## Garment session (later, same rig and settings)
### Prepare
- [ ] **Backdrop behind the whole garment**, filling the frame behind it (at ~0.7 m behind the axis, portrait framing, about 1.5 × 1.0 m): lit from behind and switchable on its own (the `backdrop` channel), for one **silhouette** frame per step. The camera and the studio never move while the garment turns, so anything static in view looks the same in every photo: without masks, COLMAP's default settings failed outright on a textured static background (synthetic, 2026-09-28), and Gaussian Wrapping would turn it into floaters.
  - Alternative: a **matte black** (velvet) backdrop and `"mask_source": "sfm"` in the plan (no extra frame; masks from the SfM image). It fails on dark garments, which disappear against it.
- [ ] **Static exclusion mask**: paint once per rig, on one developed SfM image, white over anything static in front of the backdrop (the turntable's base, a stand, cables), black elsewhere. Save it as a PNG of the same size (`static_exclude.png`). The turntable **top** turns with the garment and stays in.
- [ ] **Tape measurements** (write them down): the lens axis above the turntable top (±5 mm; puts the origin on the turntable top), and optionally the camera's entrance pupil to the turntable axis (checked within 1 %).
- [ ] **Bare-mannequin reference, once per mannequin:** a garment session with the mannequin undressed (same rig, same turntable, same camera height), processed like any garment up to GW. Its raw mesh is the reference `stage_garment.py` uses to tell the garment from the mannequin and the turntable top. Mount the mannequin the same way each time; its angle doesn't matter (the reference is registered), but its pose and any removable arms or head must match. Re-scan it if the mannequin or its stand changes.

### Capture
- `capture_session.py --plan garment`:
  1. **Metric board first**: the ChArUco board **upright on the turntable, facing the camera**, next to the mannequin (before dressing it). The script turns the table to `metric_board_angles` (default −40, −20, 0, 20, 40°) and shoots each with the all-lights shutter. Don't touch the board in between: the frames fix the turntable axis and the scale. A flat board is unreadable edge-on or from behind, so the frames stay within ±40°.
  2. Dress the mannequin. Per turntable step: the SfM image (all-lights shutter), the backlit silhouette (backdrop only, PS shutter; the backdrop may clip), ambient, each PS light alone (PS shutter), crossed: two shutter changes per step, 72 per 36-step garment. If the glTF material needs sheen or roughness, set `"parallel": "second_pass"` to add each PS light **parallel**.
- 36 steps of 10° is the plan: the verdict groups views into 30° turntable bins, so finer steps don't weaken its consistency test (tests 3 and 5 run at 12 × 30° and 36 × 10°).
- **Dark garments** (black, navy, dark batik; albedo below about 0.06): add `"ps_brackets"` to the garment plan, shutters 4× and 16× the PS shutter (e.g. `["2", "8"]` over 0.5 s). Each step then also shoots the ambient frame and every PS light at each bracket; stage 4 merges them. **Leave the PS shutter at the pilot's** (E and the colour correction were measured there). Each bracket adds 9 frames per step at a longer shutter, so a 36-step session grows by roughly 650 frames for two brackets. Why: the baked normals' per-pixel error passes 1° below about albedo 0.06 at one exposure (synthetic, CLAUDE.md → Decisions → exposure brackets); after the pilot, `bench_dark_fabric.py --noise radiometry.json` gives the real threshold.

### After (processing)
`capture_session.py --develop` (SfM images) → `stage_masks.py --session S --exclude static_exclude.png` → `stage_sfm.py --session S --intrinsics <pilot intrinsics.json> --board board.json --camera-height-m H --out R/sfm` (COLMAP with the pilot's intrinsics fixed, turntable frame in metres, gates, undistorted images, masks and lit frames) → `stage1_gw_prep.py --undistorted R/sfm/undistorted --lit-manifest R/sfm/lit/manifest.csv --masks R/sfm/undistorted/masks --out R/gw` → GW on the RTX machine → `stage2_frame_gate.py` → stage 3 → `stage_garment.py --mesh <raw GW mesh> --reference <bare-mannequin mesh> --out R/garment_faces.npy` (check `garment_faces.ply` in a mesh viewer: garment red) → stages 4–7 (give `stage7_verdict.py` stage 5's inputs, `--sparse --stage3 --stage4`, for each cluster's implied height in mm) → on BAKE, `stage_bake.py` (50k faces by default; `--max-mb 10` for Scene Viewer's recommended size). In `verdict.md`, check the board coverage line: a garment height no board placement covered is a height where the vertical blind spot is unchecked.
