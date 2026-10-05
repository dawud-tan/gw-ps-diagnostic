"""BAKE: decimated garment mesh with PS normals (and albedo) baked into textures -> .glb.

Garment faces of the raw GW mesh -> quadric decimation (to --target-faces, 50k by default: Scene Viewer
takes <= 100k triangles, and a GW mesh has millions) -> xatlas UV atlas ->
per texel, PS normals from every view that sees it (visibility by the stage-3 depth map,
weight = PS confidence x cos(view angle)) -> tangent-space normalTexture; albedo maps from
stage_albedo (optional) -> sRGB baseColorTexture. The asset is rotated from the pipeline's
world (Z up, turntable axis) to glTF's +Y up. Writes <out>.glb and <out>.json (coverage etc.).

Size: Android Scene Viewer takes a GLB of at most 15 MB (10 MB recommended). If the file with
PNG textures is bigger than --max-mb (15 by default), the base colour is stored as JPEG at falling
quality until it fits; the normal map stays PNG (bake.write_glb_within). <out>.json records the
size, how the base colour was stored, and whether it fit; exit code 1 if it did not.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh

from gwps import bake
from gwps.albedo import srgb8
from gwps.camera import load_model

Z_UP_TO_Y_UP = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float)     # (x, y, z) -> (x, z, -y)


def run(sparse, mesh, garment_faces, stage3, stage4, out, albedo=None, size=2048, keep=None, roughness=0.8,
        sheen=None, target_faces=50_000, max_mb=bake.SCENE_VIEWER_MAX_MB):
    cams, images = load_model(sparse)
    m = trimesh.load(mesh, process=False)
    garment = np.load(garment_faces).astype(bool)
    sub = bake.garment_submesh(m, garment)
    dec = bake.decimate(sub, keep) if keep is not None else bake.decimate(sub, face_count=target_faces)
    V, F, UV, N = bake.unwrap(dec)
    T4 = bake.tangents(V, F, UV, N)
    fid, (ii, jj), f, bary = bake.rasterize_atlas(F, UV, size)
    P, n, t, b = bake.texel_frames(V, F, N, T4, f, bary)
    acc_n, acc_w = np.zeros_like(P), np.zeros(len(P))
    acc_a = np.zeros_like(P) if albedo else None
    for im in images.values():
        s3 = np.load(Path(stage3) / f"{im.stem}.npz")
        s4 = np.load(Path(stage4) / f"{im.stem}.npz")
        alb = np.load(Path(albedo) / f"{im.stem}.npz")["albedo_rgb"] if albedo else None
        bake.accumulate_view(P, n, cams[im.camera_id], im.R, im.t, s3["depth"], s4["n_ps"], s4["conf"],
                             s4["ok"] & s3["hit"], acc_n, acc_w, alb, acc_a)
    seen = acc_w > 0
    n_world = np.where(seen[:, None], acc_n / np.maximum(np.linalg.norm(acc_n, axis=1, keepdims=True), 1e-12), n)
    rgb, _ = bake.encode_normals(n_world, n, t, b)
    nimg = np.zeros((size, size, 3), np.uint8)
    nimg[..., :] = (128, 128, 255)
    nimg[ii, jj] = rgb
    filled = np.zeros((size, size), bool)
    filled[ii, jj] = True
    nimg = bake.pad(nimg, filled)
    if albedo:
        a = np.where(seen[:, None], acc_a / np.maximum(acc_w, 1e-12)[:, None], np.nan)
        fallback = np.nanmedian(a, axis=0) if seen.any() else np.array([0.5, 0.5, 0.5])
        a = np.where(np.isnan(a), fallback, a)
        aimg = np.zeros((size, size, 3), np.uint8)
        aimg[..., :] = srgb8(fallback)
        aimg[ii, jj] = srgb8(a)
        aimg = bake.pad(aimg, filled)
    else:
        aimg = np.full((size, size, 3), 188, np.uint8)
    Vy = V @ Z_UP_TO_Y_UP.T
    Ny = N @ Z_UP_TO_Y_UP.T
    T4y = np.c_[T4[:, :3] @ Z_UP_TO_Y_UP.T, T4[:, 3]]
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    glb = bake.write_glb_within(out.with_suffix(".glb"), Vy, Ny, T4y, UV, F, aimg, nimg, max_mb, roughness, sheen)
    rep = {"faces_garment": int(garment.sum()), "faces_decimated": int(len(F)), "vertices_atlas": int(len(V)),
           "texture_size": size, "texels_in_atlas": int(len(P)), "texel_coverage_by_views": float(seen.mean()),
           "views": len(images), "albedo": bool(albedo), "roughness": roughness, "sheen": sheen,
           "up_axis": "glTF +Y (pipeline Z up rotated)", "glb": glb}
    out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
    cv2.imwrite(str(out.with_name(out.stem + "_normal.png")), nimg[..., ::-1])
    cv2.imwrite(str(out.with_name(out.stem + "_basecolor.png")), aimg[..., ::-1])
    return rep, (P, n, t, b, ii, jj, nimg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("sparse", "mesh", "garment-faces", "stage3", "stage4", "out"):
        ap.add_argument(f"--{k}", required=True)
    ap.add_argument("--albedo", default=None, help="stage_albedo apply output directory")
    ap.add_argument("--texture", type=int, default=2048)
    ap.add_argument("--target-faces", type=int, default=50_000, help="faces after decimation (Scene Viewer: <= 100k)")
    ap.add_argument("--keep", type=float, default=None, help="keep this fraction of faces instead of --target-faces")
    ap.add_argument("--roughness", type=float, default=0.8)
    ap.add_argument("--sheen-color", type=float, nargs=3, default=None)
    ap.add_argument("--sheen-roughness", type=float, default=0.5)
    ap.add_argument("--max-mb", type=float, default=bake.SCENE_VIEWER_MAX_MB,
                    help="GLB size budget; above it the base colour becomes JPEG (Scene Viewer: 15, 10 recommended)")
    a = ap.parse_args()
    sheen = {"color": a.sheen_color, "roughness": a.sheen_roughness} if a.sheen_color else None
    rep, _ = run(a.sparse, a.mesh, a.garment_faces, a.stage3, a.stage4, a.out, a.albedo, a.texture, a.keep,
                 a.roughness, sheen, a.target_faces, a.max_mb)
    print(json.dumps(rep, indent=1))
    if not rep["glb"]["within"]:
        print(f"GLB is {rep['glb']['bytes'] / 1e6:.1f} MB, over the {a.max_mb} MB budget even with the base colour "
              "as JPEG q75: lower --texture or --target-faces", file=sys.stderr)
        sys.exit(1)
