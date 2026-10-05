"""Check drafts/render_mesh_normals.py against an exact ray-cast reference.

Needs:  pip install bpy trimesh embreex OpenEXR scipy
(bpy = Blender as a Python module; results in README.md came from bpy 5.0.1)

What it does
 1. Builds a sphere and a COLMAP camera whose principal point is 80 px right
    of and 40 px above the image centre.
 2. Runs the draft's own import_mesh / setup_camera / render_view, unchanged.
    On Blender >= 5 this crashes (scene.node_tree no longer exists).
 3. Renders the Normal and Depth passes to a multilayer EXR instead (skipping
    the broken compositor code), with the draft's shift_x and then with the
    sign flipped, and compares each against an embree ray cast done directly
    in the COLMAP convention: silhouette IoU, and which frame the Normal pass
    is in (world / OpenCV camera / Blender camera).
"""
import importlib.util
import json
import traceback
from pathlib import Path

import numpy as np
import OpenEXR
import trimesh
from scipy.ndimage import binary_erosion
from trimesh.ray.ray_pyembree import RayMeshIntersector

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parents[1] / "drafts" / "render_mesh_normals.py"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------- scene
center = np.array([0.10, -0.05, 0.02]); radius = 0.30
sphere = trimesh.creation.icosphere(subdivisions=6, radius=radius)
sphere.apply_translation(center)
ply = OUT / "sphere.ply"; sphere.export(ply)

W, H, fx, fy, cx, cy = 640, 480, 500.0, 500.0, 400.0, 200.0
C = np.array([0.0, -2.0, 0.3])
f = (center - C) / np.linalg.norm(center - C)
x_c = np.cross(f, [0, 0, 1.0]); x_c /= np.linalg.norm(x_c)
y_c = np.cross(f, x_c)
R_cw = np.stack([x_c, y_c, f])                     # world -> camera, COLMAP
cam = {"name": "view0", "width": W, "height": H, "fx": fx, "fy": fy, "cx": cx, "cy": cy,
       "R": R_cw.tolist(), "t": (-R_cw @ C).tolist()}

# ---------------------------------------------------------------- reference ray cast
jj, ii = np.meshgrid(np.arange(W), np.arange(H))
d_cam = np.stack([(jj + 0.5 - cx) / fx, (ii + 0.5 - cy) / fy, np.ones(jj.shape)], -1).reshape(-1, 3)
d_w = d_cam / np.linalg.norm(d_cam, axis=1, keepdims=True) @ R_cw
tri, ray, loc = RayMeshIntersector(sphere).intersects_id(
    np.repeat(C[None], len(d_w), 0), d_w, multiple_hits=False, return_locations=True)
ref_mask = np.zeros(H * W, bool); ref_mask[ray] = True; ref_mask = ref_mask.reshape(H, W)
ref_n = np.zeros((H * W, 3)); ref_n[ray] = (loc - center) / radius; ref_n = ref_n.reshape(H, W, 3)
ref_z = np.zeros(H * W); ref_z[ray] = ((loc - C) @ R_cw.T)[:, 2]; ref_z = ref_z.reshape(H, W)

# ---------------------------------------------------------------- the draft, unchanged
spec = importlib.util.spec_from_file_location("draft", DRAFT)
draft = importlib.util.module_from_spec(spec); spec.loader.exec_module(draft)
import bpy  # noqa: E402  (imported by the draft already)

results = {"blender_version": bpy.app.version_string}
try:
    draft.clean_scene(); draft.import_mesh(str(ply)); draft.setup_camera(cam)
    draft.render_view(OUT, "view0")
    results["draft_render_view_as_is"] = "ran"
except Exception as e:
    results["draft_render_view_as_is"] = f"crashed: {type(e).__name__}: {e}"
    traceback.print_exc()


def render_passes(flip_shift_x):
    draft.clean_scene()
    obj = draft.import_mesh(str(ply))
    v = np.array([tuple(obj.matrix_world @ p.co) for p in obj.data.vertices])
    draft.setup_camera(cam)
    if flip_shift_x:
        bpy.context.scene.camera.data.shift_x = -(cx - W / 2) / W
    s = bpy.context.scene
    s.render.engine = "CYCLES"; s.cycles.samples = 4
    s.view_layers[0].use_pass_normal = True; s.view_layers[0].use_pass_z = True
    st = s.render.image_settings
    if "media_type" in st.bl_rna.properties.keys():        # Blender >= 5
        st.media_type = "MULTI_LAYER_IMAGE"
    st.file_format = "OPEN_EXR_MULTILAYER"; st.color_depth = "32"
    s.render.filepath = str(OUT / f"passes_flip{int(flip_shift_x)}.exr")
    bpy.ops.render.render(write_still=True)
    ch = {}
    for part in OpenEXR.File(s.render.filepath).parts:
        ch.update(part.channels)
    N = np.stack([np.asarray(ch[f"ViewLayer.Normal.{a}"].pixels, float) for a in "XYZ"], -1)
    Z = np.asarray(ch["ViewLayer.Depth.Z"].pixels, float)
    return N, Z, v


def ang(a, b):
    return float(np.degrees(np.arccos(np.clip((a * b).sum(1), -1, 1))).mean())


for flip in (False, True):
    N, Z, verts = render_passes(flip)
    key = "shift_x_sign_flipped" if flip else "shift_x_as_in_draft"
    bl_mask = np.linalg.norm(N, axis=-1) > 0.5
    iou = (bl_mask & ref_mask).sum() / (bl_mask | ref_mask).sum()
    ys, xs = np.nonzero(bl_mask)
    r = {"silhouette_IoU_vs_COLMAP_reference": round(float(iou), 3),
         "silhouette_centroid_px": [round(xs.mean(), 1), round(ys.mean(), 1)],
         "reference_centroid_px": [round(float(np.nonzero(ref_mask)[1].mean()), 1),
                                   round(float(np.nonzero(ref_mask)[0].mean()), 1)]}
    m = binary_erosion(bl_mask & ref_mask, iterations=3)
    if m.sum() > 100:
        nb = N[m] / np.linalg.norm(N[m], axis=1, keepdims=True)
        nw = ref_n[m]; ncv = nw @ R_cw.T
        r["normal_pass_mean_err_deg"] = {"if_world_frame": round(ang(nb, nw), 2),
                                         "if_opencv_camera_frame": round(ang(nb, ncv), 2),
                                         "if_blender_camera_frame": round(ang(nb, ncv * [1, -1, -1]), 2)}
        r["depth_pass_mean_abs_err_vs_planar_z_m"] = round(float(np.abs(Z[m] - ref_z[m]).mean()), 5)
    results[key] = r
results["ply_import_keeps_coordinates"] = bool(
    np.allclose(verts.min(0), center - radius, atol=1e-3) and np.allclose(verts.max(0), center + radius, atol=1e-3))

print(json.dumps(results, indent=1))
(OUT / "check_blender_renderer.json").write_text(json.dumps(results, indent=1))
