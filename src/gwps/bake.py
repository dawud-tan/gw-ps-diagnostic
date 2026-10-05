"""BAKE path: garment mesh -> decimated -> UV atlas -> PS normals (and albedo) from all views
baked into tangent-space normal and base-colour textures -> glTF 2.0 binary (.glb).

Conventions (glTF 2.0): UV (0,0) is the upper-left corner of the image (v runs down the
rows); TANGENT is exported (xyz normalised, w = +-1) and bitangent = cross(normal, tangent) * w;
the normal texture is +X right, +Y up. So T follows +u, and w makes B point along -dP/dv (up
in the image). Baking uses exactly the frame a viewer reconstructs from these attributes, so
n_world = T x + B y + N z holds for every texel.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path

import cv2
import numpy as np
import trimesh


def garment_submesh(mesh, garment):
    return mesh.submesh([np.flatnonzero(garment)], append=True)


def decimate(mesh, keep=None, face_count=None):
    """Quadric decimation to ~face_count faces, or keeping ~`keep` of them. Delivery wants a count:
    Android's Scene Viewer takes <= 100k triangles (30-50k ideal), and a raw GW mesh has millions
    of faces, so a ratio such as 0.3 would still leave millions."""
    if face_count is not None:
        return mesh if face_count >= len(mesh.faces) else mesh.simplify_quadric_decimation(face_count=int(face_count))
    return mesh.simplify_quadric_decimation(percent=1.0 - keep)


def unwrap(mesh):
    """xatlas UV atlas -> (V, F, UV, N) with per-atlas-vertex positions, normals and UVs."""
    import xatlas
    vmap, idx, uv = xatlas.parametrize(mesh.vertices.astype(np.float32), mesh.faces.astype(np.uint32))
    return (np.asarray(mesh.vertices)[vmap], idx.astype(np.int64), uv.astype(np.float64),
            np.asarray(mesh.vertex_normals)[vmap])


def tangents(V, F, UV, N):
    """Per-vertex tangents (xyz, w) in the glTF convention described above."""
    p0, p1, p2 = V[F[:, 0]], V[F[:, 1]], V[F[:, 2]]
    t0, t1, t2 = UV[F[:, 0]], UV[F[:, 1]], UV[F[:, 2]]
    e1, e2 = p1 - p0, p2 - p0
    d1, d2 = t1 - t0, t2 - t0
    det = d1[:, 0] * d2[:, 1] - d2[:, 0] * d1[:, 1]
    r = np.where(np.abs(det) > 1e-20, 1.0 / np.where(det == 0, 1, det), 0.0)
    dPdu = (e1 * d2[:, 1:2] - e2 * d1[:, 1:2]) * r[:, None]
    dPdv = (e2 * d1[:, 0:1] - e1 * d2[:, 0:1]) * r[:, None]
    Tu, Tv = np.zeros_like(V), np.zeros_like(V)
    for k in range(3):
        np.add.at(Tu, F[:, k], dPdu)
        np.add.at(Tv, F[:, k], dPdv)
    n = N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    t = Tu - np.sum(Tu * n, 1, keepdims=True) * n
    bad = np.linalg.norm(t, axis=1) < 1e-12
    if bad.any():                                         # any perpendicular direction will do
        a = np.where(np.abs(n[bad, 0:1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]])
        t[bad] = a - np.sum(a * n[bad], 1, keepdims=True) * n[bad]
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    w = np.where(np.sum(np.cross(n, t) * -Tv, 1) >= 0, 1.0, -1.0)
    return np.c_[t, w]


def rasterize_atlas(F, UV, size):
    """Texel -> face id (-1 empty) and barycentric coordinates, texel centres ((j+.5)/T, (i+.5)/T)."""
    fid = np.full((size, size), -1, np.int32)
    shift = 4
    pts = np.round((UV * size - 0.5) * (1 << shift)).astype(np.int32)
    for f, tri in enumerate(F):
        cv2.fillConvexPoly(fid, pts[tri], f, lineType=cv2.LINE_8, shift=shift)
    ii, jj = np.nonzero(fid >= 0)
    f = fid[ii, jj]
    p = np.stack([(jj + 0.5) / size, (ii + 0.5) / size], -1)
    a, b, c = UV[F[f, 0]], UV[F[f, 1]], UV[F[f, 2]]
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(1), (v0 * v1).sum(1), (v1 * v1).sum(1)
    d20, d21 = (v2 * v0).sum(1), (v2 * v1).sum(1)
    den = np.where(np.abs(d00 * d11 - d01 * d01) > 1e-30, d00 * d11 - d01 * d01, 1e-30)
    l1 = (d11 * d20 - d01 * d21) / den
    l2 = (d00 * d21 - d01 * d20) / den
    bary = np.clip(np.stack([1 - l1 - l2, l1, l2], -1), 0, 1)
    bary /= bary.sum(1, keepdims=True)
    return fid, (ii, jj), f, bary


def texel_frames(V, F, N, T4, f, bary):
    """Texel positions and the interpolated (N, T, B) frames a viewer reconstructs."""
    P = np.einsum("nk,nkc->nc", bary, V[F[f]])
    n = np.einsum("nk,nkc->nc", bary, N[F[f]])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    t = np.einsum("nk,nkc->nc", bary, T4[F[f], :3])
    t -= np.sum(t * n, 1, keepdims=True) * n
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    w = T4[F[f, 0], 3]
    b = np.cross(n, t) * w[:, None]
    return P, n, t, b


def _bilinear(img, uv):
    """Samples (N[,C]) of an (H,W[,C]) image at COLMAP-convention (u, v)."""
    x = np.clip(uv[:, 0] - 0.5, 0, img.shape[1] - 1.001)
    y = np.clip(uv[:, 1] - 0.5, 0, img.shape[0] - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    fx, fy = x - x0, y - y0
    if img.ndim == 3:
        fx, fy = fx[:, None], fy[:, None]
    return (img[y0, x0] * (1 - fx) * (1 - fy) + img[y0, x0 + 1] * fx * (1 - fy)
            + img[y0 + 1, x0] * (1 - fx) * fy + img[y0 + 1, x0 + 1] * fx * fy)


def accumulate_view(P, n_mesh, cam, R, t, depth, n_ps, conf, ok, acc_n, acc_w, albedo=None, acc_a=None,
                    depth_tol=0.004, max_view_deg=75.0):
    """Add one view's PS normals (camera frame) at the texel points P (world) to the running sums."""
    Xc = P @ R.T + t
    front = Xc[:, 2] > 1e-6
    uv = np.full((len(P), 2), -1.0)
    uv[front] = cam.project(Xc[front])
    H, W = depth.shape
    inside = front & (uv[:, 0] >= 1) & (uv[:, 0] < W - 1) & (uv[:, 1] >= 1) & (uv[:, 1] < H - 1)
    idx = np.flatnonzero(inside)
    if not len(idx):
        return
    u = uv[idx]
    zi = depth[np.clip(u[:, 1].astype(int), 0, H - 1), np.clip(u[:, 0].astype(int), 0, W - 1)]
    vis = np.abs(zi - Xc[idx, 2]) < depth_tol + 0.002 * Xc[idx, 2]
    C = -R.T @ t
    vdir = C - P[idx]
    vdir /= np.linalg.norm(vdir, axis=1, keepdims=True)
    cosv = np.sum(n_mesh[idx] * vdir, 1)
    vis &= cosv > np.cos(np.radians(max_view_deg))
    pix_ok = ok[np.clip(u[:, 1].astype(int), 0, H - 1), np.clip(u[:, 0].astype(int), 0, W - 1)]
    vis &= pix_ok
    idx, u, cosv = idx[vis], u[vis], cosv[vis]
    if not len(idx):
        return
    nc = _bilinear(n_ps, u)
    nw = nc @ R                                            # camera -> world
    nw /= np.maximum(np.linalg.norm(nw, axis=1, keepdims=True), 1e-12)
    w = _bilinear(conf, u) * cosv
    np.add.at(acc_n, idx, nw * w[:, None])
    np.add.at(acc_w, idx, w)
    if albedo is not None and acc_a is not None:
        np.add.at(acc_a, idx, _bilinear(albedo, u) * w[:, None])


def encode_normals(n_world, n, t, b):
    """World normals -> tangent space (x along T, y along B, z along N) -> 8-bit RGB."""
    ts = np.stack([np.sum(n_world * t, 1), np.sum(n_world * b, 1), np.sum(n_world * n, 1)], -1)
    ts /= np.maximum(np.linalg.norm(ts, axis=1, keepdims=True), 1e-12)
    return np.round((ts * 0.5 + 0.5) * 255).astype(np.uint8), ts


def decode_normals(rgb8, n, t, b):
    ts = rgb8.astype(np.float64) / 255 * 2 - 1
    ts /= np.maximum(np.linalg.norm(ts, axis=1, keepdims=True), 1e-12)
    return ts[:, 0:1] * t + ts[:, 1:2] * b + ts[:, 2:3] * n


def pad(img, filled, iters=8):
    """Bleed filled texels into empty ones (mip-map and bilinear safety at chart seams)."""
    img, filled = img.copy(), filled.copy()
    k = np.ones((3, 3), np.uint8)
    for _ in range(iters):
        grown = cv2.dilate(filled.astype(np.uint8), k) > 0
        ring = grown & ~filled
        if not ring.any():
            break
        f = filled.astype(np.float32)
        s = cv2.blur(img.astype(np.float32) * f[..., None], (3, 3)) / np.maximum(cv2.blur(f, (3, 3)), 1e-6)[..., None]
        img[ring] = np.round(s[ring]).astype(img.dtype)
        filled = grown
    return img


# ---------------------------------------------------------------- glTF 2.0 binary writer
def write_glb(path, V, N, T4, UV, F, base_png, normal_png, roughness=0.8, sheen=None, name="garment",
              base_mime="image/png"):
    """Minimal, spec-conformant GLB: one mesh, one PBR material (metallic 0), baseColorTexture
    (sRGB PNG, or JPEG with base_mime="image/jpeg"), normalTexture (tangent space, PNG), explicit
    TANGENT, optional KHR_materials_sheen."""
    blobs, views, accessors = [], [], []

    def add_view(data, target=None):
        off = sum(len(b) for b in blobs)
        pad_ = (-off) % 4
        if pad_:
            blobs.append(b"\x00" * pad_)
            off += pad_
        blobs.append(data)
        v = {"buffer": 0, "byteOffset": off, "byteLength": len(data)}
        if target:
            v["target"] = target
        views.append(v)
        return len(views) - 1

    def add_accessor(arr, ctype, typ, target, minmax=False):
        a = np.ascontiguousarray(arr)
        vi = add_view(a.tobytes(), target)
        acc = {"bufferView": vi, "componentType": ctype, "count": int(a.shape[0]), "type": typ}
        if minmax:
            acc["min"], acc["max"] = a.min(0).tolist(), a.max(0).tolist()
        accessors.append(acc)
        return len(accessors) - 1

    FLOAT, UINT = 5126, 5125
    ARRAY, ELEMENTS = 34962, 34963
    pos = add_accessor(V.astype(np.float32), FLOAT, "VEC3", ARRAY, minmax=True)
    nrm = add_accessor((N / np.linalg.norm(N, axis=1, keepdims=True)).astype(np.float32), FLOAT, "VEC3", ARRAY)
    tan = add_accessor(T4.astype(np.float32), FLOAT, "VEC4", ARRAY)
    tex = add_accessor(UV.astype(np.float32), FLOAT, "VEC2", ARRAY)
    ind = add_accessor(F.astype(np.uint32).ravel(), UINT, "SCALAR", ELEMENTS)
    img_base = add_view(base_png)
    img_norm = add_view(normal_png)
    material = {"name": f"{name}_material",
                "pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0.0,
                                         "roughnessFactor": float(roughness)},
                "normalTexture": {"index": 1}}
    gltf = {"asset": {"version": "2.0", "generator": "gw-ps-diagnostic bake"},
            "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": name}],
            "meshes": [{"name": name, "primitives": [{"attributes": {"POSITION": pos, "NORMAL": nrm, "TANGENT": tan,
                                                                     "TEXCOORD_0": tex},
                                                      "indices": ind, "material": 0}]}],
            "materials": [material],
            "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
            "images": [{"bufferView": img_base, "mimeType": base_mime}, {"bufferView": img_norm, "mimeType": "image/png"}],
            "textures": [{"sampler": 0, "source": 0}, {"sampler": 0, "source": 1}],
            "accessors": accessors, "bufferViews": views}
    if sheen:
        material["extensions"] = {"KHR_materials_sheen": {"sheenColorFactor": list(map(float, sheen["color"])),
                                                          "sheenRoughnessFactor": float(sheen["roughness"])}}
        gltf["extensionsUsed"] = ["KHR_materials_sheen"]
    binary = b"".join(blobs)
    binary += b"\x00" * ((-len(binary)) % 4)
    gltf["buffers"] = [{"byteLength": len(binary)}]
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<III", 0x46546C67, 2, total))
        fh.write(struct.pack("<II", len(js), 0x4E4F534A) + js)
        fh.write(struct.pack("<II", len(binary), 0x004E4942) + binary)
    return path


# Android Scene Viewer takes a GLB of at most 15 MB (10 MB recommended; CLAUDE.md, delivery limits).
SCENE_VIEWER_MAX_MB = 15.0
JPEG_QUALITIES = (95, 90, 85, 80, 75)


def write_glb_within(path, V, N, T4, UV, F, base_rgb, normal_rgb, max_mb=SCENE_VIEWER_MAX_MB, roughness=0.8,
                     sheen=None, name="garment"):
    """write_glb with both textures as PNG; if the file exceeds max_mb, the base colour again as
    JPEG at falling quality (JPEG_QUALITIES) until it fits. The normal map always stays PNG: JPEG's
    8 x 8 blocks and chroma subsampling would be read as tilted normals, and the base colour is
    what a viewer forgives. base_rgb / normal_rgb: (H,W,3) uint8 RGB. Returns {"bytes", "max_bytes",
    "base_colour" ("png" or "jpeg q<N>"), "within": bool}; within is False if even the lowest
    quality is too big (the file is still written, at that quality, and the caller must say so)."""
    path = Path(path)
    limit = int(max_mb * 1e6)
    ok, png_n = cv2.imencode(".png", normal_rgb[..., ::-1])
    ok2, png_a = cv2.imencode(".png", base_rgb[..., ::-1])
    if not (ok and ok2):
        raise RuntimeError("texture encoding failed")
    write_glb(path, V, N, T4, UV, F, png_a.tobytes(), png_n.tobytes(), roughness, sheen, name)
    size, how = path.stat().st_size, "png"
    for q in JPEG_QUALITIES:
        if size <= limit:
            break
        ok, jpg = cv2.imencode(".jpg", base_rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])
        write_glb(path, V, N, T4, UV, F, jpg.tobytes(), png_n.tobytes(), roughness, sheen, name, base_mime="image/jpeg")
        size, how = path.stat().st_size, f"jpeg q{q}"
    return {"bytes": int(size), "max_bytes": limit, "base_colour": how, "within": size <= limit}
