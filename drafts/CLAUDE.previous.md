# CLAUDE.md

## Task
Diagnostic tool to decide whether Gaussian Wrapping needs PS-normal
training supervision built (novel, unpublished per prior-art check) or
whether baking a PS normal map onto its output mesh is sufficient.
Per the prior-art check's own recommendation: measure disagreement
between GW's mesh and independently-solved photometric-stereo normals,
separated into low/mid-frequency (real shape gap -> worth building) vs
high-frequency (mesh-resolution-limited -> baking is just as good).

## Capture rig
Garment on mannequin, 360deg turntable, fixed viewpoint per step,
varying known lighting per step (photometric stereo pattern). COLMAP
sparse reconstruction feeds directly into Gaussian Wrapping, which
expects a COLMAP-formatted dataset (images + sparse/) natively.

## Gaussian Wrapping
Repo: github.com/diego1401/GaussianWrapping
Use train_and_extract_gw_radegs.py (best-looking meshes; the faster
"ours" rasterizer is tuned for benchmark metrics, not visual quality).
--isosurface_value 0.2 if fine detail is missing. Skip their benchmark
post-processing for anything getting textured -- it can delete real
thin geometry. Decimate (Blender, ratio 0.3) before texturing.

## Photometric stereo conventions
Light directions in lights.json are CAMERA space, OpenCV/COLMAP
convention (X right, Y down, Z forward) -- same as the camera poses.
Calibration tools often use a different convention (Y-up); get this
wrong and it's silently mirrored, not obviously broken.
Grazing-angle cutoff at 75deg from view direction is validated, not a
guess -- confirmed empirically against synthetic ground truth to
roughly halve mean angular error near silhouettes.

## Known gotcha -- do not skip
trimesh's default `mesh.ray` (pure-Python ray_triangle) OOMs a 3.9GB
container at under 10,000 rays against a 5,000-face mesh. Must use
`pip install embreex` and `trimesh.ray.ray_pyembree.RayMeshIntersector`
explicitly. Confirmed working: 1M rays against a 327k-face mesh in
~0.35s / 460MB with embreex, vs OOM well before that with the default.

## Current state
photometric_stereo.py -- complete, validated against synthetic ground
truth (near-zero error in well-conditioned regions; error grows toward
grazing angles as expected, not a bug).
gw_ps_diagnostic.py -- embree fix applied; a blank-POINTS2D-line parser
bug in read_colmap_images was just fixed. Re-run the synthetic
integration test in synth_test/ to confirm both viewpoints process
cleanly before touching real data.