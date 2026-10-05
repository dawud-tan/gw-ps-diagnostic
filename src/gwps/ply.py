"""Heatmap PLY: mesh with one float scalar per face (plus a colour for quick viewing)."""
import numpy as np


def write_face_scalar_ply(path, vertices, faces, scalar, name="theta_deg", vmax=None):
    s = np.asarray(scalar, np.float32)
    fin = np.isfinite(s)
    vmax = vmax or (float(np.nanpercentile(s, 99)) if fin.any() else 1.0) or 1.0
    x = np.clip(np.where(fin, s, 0) / vmax, 0, 1)
    rgb = np.stack([255 * x, 255 * (1 - np.abs(2 * x - 1)), 255 * (1 - x)], -1).astype(np.uint8)
    rgb[~fin] = 128
    v = np.asarray(vertices, np.float32)
    f = np.asarray(faces, np.int32)
    head = (f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            f"element face {len(f)}\nproperty list uchar int vertex_indices\nproperty float {name}\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    fdt = np.dtype([("n", "u1"), ("i", "<i4", 3), ("s", "<f4"), ("c", "u1", 3)])
    rec = np.zeros(len(f), fdt)
    rec["n"], rec["i"], rec["s"], rec["c"] = 3, f, s, rgb
    with open(path, "wb") as fh:
        fh.write(head.encode())
        fh.write(v.astype("<f4").tobytes())
        fh.write(rec.tobytes())
