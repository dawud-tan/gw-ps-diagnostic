"""
Photometric-stereo solve + frequency-separated disagreement check against a
Gaussian Wrapping mesh's own rendered normals.

This is the cheap diagnostic: if disagreement between the mesh's normals
and independently-computed PS normals is high-frequency only (finer than
the mesh could represent even in principle), bake PS normals into a UV
texture and stop. If there's real low- or mid-frequency disagreement
(coarser than the mesh's own resolution -- meaning Gaussian Wrapping is
getting something wrong at a scale it should be able to capture), that's
empirical justification for building the training-time L_PS integration.

Usage:
  python ps_disagreement_check.py \
      --lit-images <dir with step####_light#.jpg files> \
      --calibration lights.json \
      --rendered <output dir from render_mesh_normals.py> \
      --cameras <same cameras.json fed to render_mesh_normals.py> \
      --out <report dir>

lights.json format -- see lights.example.json alongside this script.
Light directions are given in the SAME world frame model_aligner produced
(metres, origin at the turntable's rotation centre), as they'd be measured
at turntable step 0. At step N, the true light-to-garment direction has
rotated by -N * degrees_per_step around rotation_axis_world relative to
that reference -- because COLMAP holds the garment fixed in its
reconstruction and represents the turntable's rotation as apparent camera
motion instead, so anything genuinely fixed in the studio (the lights)
appears to rotate backward by the same amount, every step.

Image filename convention expected: stepNNNN_lightK.jpg (e.g.
step0042_light2.jpg). If your rig names things differently, adjust
FNAME_RE below rather than renaming your captures.
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

FNAME_RE = re.compile(r"step(\d+)_light(\d+)")


def rotation_matrix(axis, angle_rad):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    C = 1 - c
    return np.array(
        [
            [x * x * C + c, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, y * y * C + c, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, z * z * C + c],
        ]
    )


def effective_light_directions(calib, step_index):
    """Undo the turntable rotation that COLMAP's reconstruction absorbed
    into apparent camera motion -- see the module docstring."""
    angle = -math.radians(step_index * calib["degrees_per_step"])
    R = rotation_matrix(calib["rotation_axis_world"], angle)
    return {
        L["id"]: R @ np.asarray(L["direction_at_step_zero"], dtype=np.float64)
        for L in calib["lights"]
    }


def solve_ps(images, light_dirs_cam):
    """images: [K, H, W] float intensities. light_dirs_cam: [K, 3] light
    directions already rotated into this camera's local frame. Assumes
    distant (directional, not point/near) lighting, i.e. one direction per
    light shared across the whole frame -- reasonable for lights a metre-plus
    from a ~0.5 m garment; revisit if that assumption doesn't hold for your
    rig. Returns unit normals [H,W,3], albedo [H,W], per-pixel fit
    residual [H,W]."""
    K, H, W = images.shape
    L = np.asarray(light_dirs_cam, dtype=np.float64)  # [K, 3]
    L_pinv = np.linalg.pinv(L)  # [3, K]

    I = images.reshape(K, H * W).astype(np.float64)  # [K, HW]
    g = L_pinv @ I  # [3, HW] -- albedo-scaled normal, linearized PS

    albedo = np.linalg.norm(g, axis=0)
    normal = g / np.clip(albedo, 1e-6, None)
    residual = np.linalg.norm(L @ g - I, axis=0) / max(K, 1)

    return (
        normal.T.reshape(H, W, 3).astype(np.float32),
        albedo.reshape(H, W).astype(np.float32),
        residual.reshape(H, W).astype(np.float32),
    )


def confidence_mask(images, residual, shadow_thresh=8, sat_thresh=250, residual_thresh=25.0):
    """images: [K, H, W] in 0-255 range. Flags shadowed pixels (near-zero
    under some light), saturated pixels (blown out), and pixels the
    Lambertian fit doesn't explain well (likely specular or interreflection
    -- worth watching closely on fabric with any sheen)."""
    not_shadowed = (images > shadow_thresh).all(axis=0)
    not_saturated = (images < sat_thresh).all(axis=0)
    good_fit = residual < residual_thresh
    return not_shadowed & not_saturated & good_fit


def angular_error_deg(n1, n2):
    dot = np.clip(np.sum(n1 * n2, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def masked_gaussian_blur(x, mask, sigma):
    sigma = max(sigma, 0.5)
    num = gaussian_filter(x * mask, sigma=sigma)
    den = gaussian_filter(mask.astype(np.float64), sigma=sigma)
    return num / np.clip(den, 1e-6, None)


def save_heatmaps(out_dir, name, theta, theta_low, theta_high, mask, vmax=30.0):
    def to_img(x):
        x = np.clip(x, 0, vmax) / vmax
        x = np.where(mask, x, 0.0)
        return Image.fromarray((x * 255).astype(np.uint8))

    to_img(theta).save(out_dir / f"{name}_disagreement_raw.png")
    to_img(theta_low).save(out_dir / f"{name}_disagreement_lowfreq.png")
    to_img(np.abs(theta_high)).save(out_dir / f"{name}_disagreement_highfreq.png")


def process_view(step_index, light_images, calib, cam_R, rendered_normal,
                  rendered_depth, median_edge_m, focal_px, out_dir, name):
    imgs = np.stack([np.asarray(im, dtype=np.float32) for im in light_images], axis=0)

    eff_world = effective_light_directions(calib, step_index)
    light_ids = sorted(eff_world.keys())
    L_cam = np.stack([np.asarray(cam_R) @ eff_world[i] for i in light_ids], axis=0)

    n_ps, albedo, residual = solve_ps(imgs, L_cam)
    conf = confidence_mask(imgs, residual)

    # "no geometry hit" convention depends on your Blender/Cycles version --
    # sanity-check this threshold against an actual depth.npy from your run
    # before trusting the mask (e.g. print np.unique on a background patch).
    valid_depth = (rendered_depth > 1e-3) & (rendered_depth < 1e4)
    mask = conf & valid_depth

    n_mesh = rendered_normal
    norm = np.linalg.norm(n_mesh, axis=-1, keepdims=True)
    n_mesh = n_mesh / np.clip(norm, 1e-6, None)

    theta = angular_error_deg(n_mesh, n_ps)

    mean_depth = float(np.median(rendered_depth[mask])) if mask.any() else float("nan")
    sigma_px = focal_px * median_edge_m / max(mean_depth, 1e-3) if mask.any() else 5.0

    theta_low = masked_gaussian_blur(theta, mask.astype(np.float64), sigma_px)
    theta_high = theta - theta_low

    rms_low = float(np.sqrt(np.mean(theta_low[mask] ** 2))) if mask.any() else float("nan")
    rms_high = float(np.sqrt(np.mean(theta_high[mask] ** 2))) if mask.any() else float("nan")

    save_heatmaps(out_dir, name, theta, theta_low, theta_high, mask)

    return {
        "name": name,
        "step_index": step_index,
        "sigma_px": sigma_px,
        "valid_fraction": float(mask.mean()),
        "rms_low_freq_deg": rms_low,
        "rms_high_freq_deg": rms_high,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lit-images", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--rendered", required=True)
    ap.add_argument("--cameras", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--low-freq-threshold-deg", type=float, default=7.0,
        help="Starting judgment call, not a derived constant -- there's no "
             "ground truth here to calibrate an exact cutoff against. Look "
             "at the *_lowfreq.png heatmaps before trusting this number.",
    )
    args = ap.parse_args()

    lit_dir = Path(args.lit_images)
    rendered_dir = Path(args.rendered)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib = json.loads(Path(args.calibration).read_text())
    cameras = {c["name"]: c for c in json.loads(Path(args.cameras).read_text())}
    median_edge_m = json.loads((rendered_dir / "mesh_stats.json").read_text())[
        "median_edge_length_m"
    ]

    by_step = {}
    for f in lit_dir.glob("*.jpg"):
        m = FNAME_RE.search(f.stem)
        if not m:
            continue
        step, light = int(m.group(1)), int(m.group(2))
        by_step.setdefault(step, {})[light] = f

    results = []
    for step_index, lights in sorted(by_step.items()):
        name = f"step{step_index:04d}"
        if name not in cameras:
            continue
        cam = cameras[name]

        normal_path = rendered_dir / f"{name}_normal.npy"
        depth_path = rendered_dir / f"{name}_depth.npy"
        if not (normal_path.exists() and depth_path.exists()):
            print(f"skipping {name}: no rendered normal/depth found")
            continue

        light_images = [Image.open(lights[i]).convert("L") for i in sorted(lights)]
        rendered_normal = np.load(normal_path)
        rendered_depth = np.load(depth_path)

        result = process_view(
            step_index, light_images, calib, np.array(cam["R"]),
            rendered_normal, rendered_depth, median_edge_m, cam["fx"], out_dir, name,
        )
        results.append(result)
        print(
            f"{name}: low={result['rms_low_freq_deg']:.2f} deg  "
            f"high={result['rms_high_freq_deg']:.2f} deg  "
            f"valid={result['valid_fraction']:.0%}"
        )

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))

    valid = [r for r in results if not math.isnan(r["rms_low_freq_deg"])]
    if not valid:
        print("\nno usable views -- check filename matching and valid_fraction above 0")
        return

    mean_low = float(np.mean([r["rms_low_freq_deg"] for r in valid]))
    mean_high = float(np.mean([r["rms_high_freq_deg"] for r in valid]))

    print(f"\n=== VERDICT (n={len(valid)} views) ===")
    print(f"mean low-frequency disagreement:  {mean_low:.2f} deg")
    print(f"mean high-frequency disagreement: {mean_high:.2f} deg")
    if mean_low < args.low_freq_threshold_deg:
        print("-> low-frequency disagreement is small: bake PS normals, skip L_PS.")
    else:
        print(
            "-> real low/mid-frequency disagreement: worth building L_PS. "
            "Inspect the *_lowfreq.png heatmaps to see WHERE before committing."
        )


if __name__ == "__main__":
    main()
