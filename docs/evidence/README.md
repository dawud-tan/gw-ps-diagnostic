# Evidence for the "Known bugs in drafts/" list in CLAUDE.md

Checked on 2026-09-25 against the files in `drafts/`. Each script is self-contained. Rerun them before relying on the numbers below. Outputs go to `out/`, which is not committed.

| Script | Needs | Checks |
|---|---|---|
| `check_bands_and_ps_bias.py` | numpy, scipy | (A) the frequency-separation method in `ps_disagreement_check.py`; (B) how much bias the photometric-stereo (PS) model and radiometry add |
| `check_blender_renderer.py` | bpy, trimesh, embreex, OpenEXR, scipy | `render_mesh_normals.py` on Blender 5.0.1: whether it runs, the principal-point shift, and which frame the Normal pass is in |
| `check_images_txt_parser.py` | numpy | `build_camera_list.py` on an `images.txt` that has an empty POINTS2D line |

Versions used: Python 3.11, numpy 1.26, scipy 1.17, bpy 5.0.1, trimesh 5.1.0 + embreex, pycolmap 4.2.0.

## Results observed

### A. Low-pass filtering the angle map fakes low-frequency disagreement
The draft blurs the per-pixel angle θ = ∠(n_mesh, n_ps). Because θ ≥ 0, fine texture rectifies into a positive local mean and shows up as low-frequency error. The table uses σ = 6 px, the draft's "one mesh edge".

| Case (true low-frequency disagreement) | Draft method, RMS low | Vector low-pass, RMS low |
|---|---|---|
| Weave-like ripple ±6° plus noise (0°) | **2.89°** | 0.09° |
| Weave-like ripple ±12° plus noise (0°) | **5.79°** | 0.19° |
| Real 8° bump, 60 px wide, no ripple | 5.73° | 5.73° |

With the draft's 7° threshold, a fabric with slightly stronger texture or PS noise produces "build L_PS" with no shape error at all. The fix is to low-pass both normal fields as vectors, renormalise them, and only then take the angle.

### B. PS model and radiometry bias, all smooth, i.e. in the band the verdict reads
The simulation uses a camera-frame garment region 0.5 × 0.9 m with ±0.1 m relief, 1.5 m from the camera. It has 6 lights at 35° off the viewing axis and normals up to 40° off-axis.

| Error source | Median | p90 / max |
|---|---|---|
| Point lights 1.0 m away, directional solve | 28.5° | 46.7° / 70.4° |
| Point lights 1.5 m away, directional solve | 20.6° | 34.3° / 48.7° |
| Point lights 2.0 m away, directional solve | 16.1° | 26.5° / 36.8° |
| Point lights 3.0 m away, directional solve | 11.1° | 18.2° / 24.5° |
| Point lights 1.5 m, **near-light solve** using positions with 5 mm error | 0.33° | max 0.51° |
| sRGB (JPEG) values used as linear | 10.7° | max 18.1° |
| Light intensities off by ±10 % | 1.0° | max 1.3° |
| Near-light solve, light positions miscalibrated by 5 mm | 0.43° | p90 0.54° |
| Near-light solve, light positions miscalibrated by 10 mm | 0.47° | p90 0.60° |
| Near-light solve, light positions miscalibrated by 20 mm | 1.26° | p90 1.84° |
| 60 cm softbox at 1.5 m, solved as a point light with emitter cosine | 1.03° | p90 1.72° |

The miscalibration offsets point in a random direction per light, so the 5 mm and 10 mm rows differ by less than you might expect. What matters is that 20 mm of error, or treating a softbox as a point, uses up most of a 1° margin.

The error grows with distance from the centre. With lights at 1.5 m, a frontal normal has 0° error at the centre, 4.7° at 5 cm, 9.3° at 10 cm, 21.8° at 25 cm and 34.5° at 45 cm. The draft's claim that a directional model is "reasonable for lights a metre-plus from a ~0.5 m garment" does not hold at this precision.

A near-light solve that takes each pixel's 3-D position from the GW mesh fixes this. Mesh position errors of a few mm barely move the light vectors, so no GW error leaks into the PS normals.

### C. `render_mesh_normals.py` on Blender 5.0.1
- **It crashes as-is:** `AttributeError: 'Scene' object has no attribute 'node_tree'`. The compositor API changed in Blender 5.
- **The `shift_x` sign is wrong.** The test puts the principal point 80 px right of centre. The draft's render lands at x = 239.5 while COLMAP projects to x = 399.5, a 160 px offset with silhouette IoU 0.00. With the sign flipped, IoU is 0.998. `shift_y` is correct.
- **The Cycles Normal pass is in world space.** Mean error is 0.58° against the world frame, 85.3° against the OpenCV camera frame and 71.8° against the Blender camera frame. The draft's docstring says camera space, and `ps_disagreement_check.py` compares it directly with camera-frame PS normals.
- The Depth pass is planar z, with mean absolute error 0.0 m against planar depth.
- The PLY import keeps coordinates, so no axis conversion is needed.

### D. `build_camera_list.py` with an empty POINTS2D line
The draft parser silently loses `step0002` and invents an image named `70` with camera_id 5 and t = [-1, 50, 60]. The fixed parser in the script keeps blank lines and reads all three images correctly.

### E. Other measured points
- `ps_disagreement_diagnostic.py` stores each per-face sample as a Python tuple, about **235 bytes per sample** (2.4 GB per 10 M samples).
  - Worse, each `normals_world[y, x]` is a numpy *view* (`v.base is a` → True), so every stored sample keeps its view's whole normal array alive: about 576 MB per 24 MP view.
  - A 3.9 GB container therefore runs out after only a few views.
- GW ignores the principal point. `scene/cameras.py` sets Cx = (W−1)/2 and Cy = (H−1)/2.
  - pycolmap 4.2 `undistort_camera`: a centred input stays centred (+0.00 px), and +40 px off-centre in becomes +41.92 px out.
  - So self-calibrated COLMAP models are fine, and externally calibrated ones with an off-centre principal point are not.
- GW's raw mesh `…_searched.ply` is always written before post-processing. Post-processing (`post_process_mesh(cluster_to_keep=1)`) keeps only the largest connected component.
- pycolmap 4.2.0: `Image.cam_from_world` is a method now, so `image.cam_from_world()`. The draft uses it as an attribute. `Rigid3d.rotation.matrix()` and `.inverse().translation` still work.
- trimesh 5.1 with embreex installed picks embree automatically for `mesh.ray`. Without embreex it silently falls back to the pure-Python intersector.
