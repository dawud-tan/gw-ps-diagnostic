"""Extract per-image camera intrinsics + extrinsics from a COLMAP sparse
model (TEXT format) into the flat JSON format render_mesh_normals.py and
ps_disagreement_check.py expect.

Reads the same cameras.txt / images.txt already produced by
`colmap model_converter --output_type TXT` in run_pipeline.sh's
registration-rate QA gate -- no pycolmap dependency, so this stays stable
across COLMAP/pycolmap version drift (pycolmap's Python API has changed
shape more than once; the TXT format has not).

Usage:
  python build_camera_list.py --sparse-txt <path/to/sparse/0> --out cameras.json

Assumes PINHOLE or OPENCV-family camera models (fx, fy, cx, cy as the
first four params). Distortion params, if present, are ignored here --
this script's output describes camera poses for rendering a mesh, and the
photos it's compared against were already undistorted upstream
(`colmap image_undistorter`) in the main pipeline, so nothing downstream
needs to apply distortion again.
"""
import argparse
import json
from pathlib import Path


def quat_to_rotmat(qw, qx, qy, qz):
    n = (qw * qw + qx * qx + qy * qy + qz * qz) ** 0.5
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]


def parse_cameras_txt(path):
    cams = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split()
        cam_id, model, w, h = int(parts[0]), parts[1], int(parts[2]), int(parts[3])
        params = [float(p) for p in parts[4:]]
        if model in ("PINHOLE", "OPENCV", "OPENCV_FISHEYE"):
            fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        elif model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"):
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:
            raise ValueError(
                f"unhandled camera model '{model}' -- add its param layout above"
            )
        cams[cam_id] = {"width": w, "height": h, "fx": fx, "fy": fy, "cx": cx, "cy": cy}
    return cams


def parse_images_txt(path):
    lines = [
        l for l in Path(path).read_text().splitlines()
        if l.strip() and not l.startswith("#")
    ]
    images = []
    # images.txt alternates: one line of pose data, one line of 2D points,
    # per image -- we only need the pose line.
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]
        images.append(
            {
                "name": Path(name).stem,  # drop extension to match the PS filename stem
                "camera_id": cam_id,
                "R": quat_to_rotmat(qw, qx, qy, qz),  # world -> camera
                "t": [tx, ty, tz],
            }
        )
    return images


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sparse-txt", required=True,
        help="directory containing cameras.txt and images.txt (TXT-format COLMAP model)",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sparse = Path(args.sparse_txt)
    cams = parse_cameras_txt(sparse / "cameras.txt")
    images = parse_images_txt(sparse / "images.txt")

    out = []
    for img in images:
        cam = cams[img["camera_id"]]
        out.append({"name": img["name"], **cam, "R": img["R"], "t": img["t"]})

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} camera entries to {args.out}")


if __name__ == "__main__":
    main()
