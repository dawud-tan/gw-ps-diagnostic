"""
Blender headless: render camera-space normal + depth for a Gaussian
Wrapping mesh, from a set of known COLMAP camera poses -- for comparing
against independently-computed photometric-stereo normals.

Run with:
  blender --background --factory-startup --python render_mesh_normals.py -- \
      --mesh <mesh.ply> --cameras <cameras.json> --out <out_dir>

cameras.json: the output of build_camera_list.py -- a list of
  {"name", "width", "height", "fx", "fy", "cx", "cy", "R" (3x3, world->camera,
  COLMAP convention: x-right, y-down, z-forward), "t" (3,)}.

IMPORTANT -- verify before trusting the output: the COLMAP-to-Blender camera
conversion below, especially the shift_x/shift_y principal-point mapping, is
the single most likely place for a silent, hard-to-notice bug. Before
running the full comparison, render one view's RGB (not just normal/depth --
swap the viewer link to a Combined pass temporarily) and eyeball it against
the real photo from that camera. If the framing lines up, the geometry is
right; if there's an offset or skew, the shift formula needs adjusting for
your Blender version.

Output per view: <name>_normal.npy (H,W,3 float32, camera-space, unit
length where valid), <name>_depth.npy (H,W float32, metres). Also writes
mesh_stats.json once, with the mesh's median edge length in metres -- the
frequency-separation step needs this to know what "finer than the mesh can
represent" means at each view.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import bpy
import numpy as np
import mathutils


def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--out", required=True)
    return ap.parse_args(argv)


def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_mesh(path):
    ext = Path(path).suffix.lower()
    try:
        if ext == ".ply":
            bpy.ops.wm.ply_import(filepath=path)
        elif ext == ".obj":
            bpy.ops.wm.obj_import(filepath=path)
        else:
            raise ValueError(f"unsupported mesh format: {ext}")
    except AttributeError:
        # older Blender: fall back to the legacy importer names
        if ext == ".ply":
            bpy.ops.import_mesh.ply(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    return next(o for o in bpy.context.scene.objects if o.type == "MESH")


def median_edge_length(obj):
    import bmesh
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    lengths = sorted(e.calc_length() for e in bm.edges)
    bm.free()
    return lengths[len(lengths) // 2] if lengths else 0.0


def colmap_to_blender_camera(cam_spec):
    """COLMAP: x-right, y-down, z-forward (into scene), world-to-camera R, t.
    Blender's camera looks down local -Z with +Y up -- the standard fix is a
    180-degree rotation about the camera's own local X axis."""
    R_cw = mathutils.Matrix(cam_spec["R"])  # world -> camera (COLMAP)
    t_cw = mathutils.Vector(cam_spec["t"])
    R_wc = R_cw.transposed()  # camera -> world
    cam_center_world = -(R_wc @ t_cw)

    flip = mathutils.Matrix.Rotation(math.pi, 3, "X")  # COLMAP cam -> Blender cam
    R_blender_c2w = R_wc @ flip

    mat = R_blender_c2w.to_4x4()
    mat.translation = cam_center_world
    return mat


def setup_camera(cam_spec):
    cam_data = bpy.data.cameras.new("diag_cam")
    cam_obj = bpy.data.objects.new("diag_cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    w, h = cam_spec["width"], cam_spec["height"]
    fx = cam_spec["fx"]
    cx, cy = cam_spec["cx"], cam_spec["cy"]

    scene = bpy.context.scene
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.resolution_percentage = 100

    cam_data.sensor_fit = "HORIZONTAL"
    cam_data.sensor_width = 36.0  # arbitrary reference value; lens below is
                                   # derived consistently against it, so the
                                   # ratio -- not the absolute number -- is
                                   # what matters
    cam_data.lens = fx * cam_data.sensor_width / w
    # Blender's shift_x/shift_y are both normalized by sensor_width when
    # sensor_fit == 'HORIZONTAL' -- this is the part flagged above as worth
    # a manual sanity check.
    cam_data.shift_x = (cx - w / 2) / w
    cam_data.shift_y = (cy - h / 2) / w

    cam_obj.matrix_world = colmap_to_blender_camera(cam_spec)
    return cam_obj


def render_pass(pass_name):
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.nodes.clear()
    rl = tree.nodes.new("CompositorNodeRLayers")
    viewer = tree.nodes.new("CompositorNodeViewer")
    tree.links.new(rl.outputs[pass_name], viewer.inputs[0])

    bpy.ops.render.render(write_still=False)

    img = bpy.data.images["Viewer Node"]
    h, w = scene.render.resolution_y, scene.render.resolution_x
    px = np.array(img.pixels[:], dtype=np.float32).reshape(h, w, 4)
    return np.flipud(px)  # Blender's pixel buffer is bottom-up


def render_view(out_dir, name):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    scene.cycles.samples = 4  # geometry-only passes: noise doesn't matter
    scene.view_layers[0].use_pass_normal = True
    scene.view_layers[0].use_pass_z = True

    normal_rgba = render_pass("Normal")
    depth_rgba = render_pass("Depth")

    np.save(out_dir / f"{name}_normal.npy", normal_rgba[..., :3].astype(np.float32))
    np.save(out_dir / f"{name}_depth.npy", depth_rgba[..., 0].astype(np.float32))


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_scene()
    obj = import_mesh(args.mesh)
    edge_len = median_edge_length(obj)
    (out_dir / "mesh_stats.json").write_text(
        json.dumps({"median_edge_length_m": edge_len})
    )

    cameras = json.loads(Path(args.cameras).read_text())
    for cam_spec in cameras:
        setup_camera(cam_spec)
        render_view(out_dir, cam_spec["name"])
        bpy.data.objects.remove(bpy.context.scene.camera, do_unlink=True)

    print(f"rendered {len(cameras)} views; median mesh edge length = {edge_len:.5f} m")


if __name__ == "__main__":
    main()
