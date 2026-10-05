"""Evidence for two issues in the drafts. Self-contained: numpy + scipy only.

(A) Frequency separation. drafts/ps_disagreement_check.py low-passes the
    per-pixel angular error map theta = angle(n_mesh, n_ps). Because theta is
    a magnitude (>= 0), high-frequency disagreement rectifies into a positive
    local mean and shows up as "low-frequency". Test: PS normals that differ
    from the mesh ONLY by a fine weave-like ripple.
    Correct alternative: low-pass both normal fields (vectors), renormalise,
    then take the angle.

(B) PS model bias (plus light-calibration and area-light error). A distant-light (directional) Lambertian solve, applied to
    a mannequin-sized garment lit by point lights at 1-3 m, mis-tilts normals
    by a smooth, position-dependent amount, i.e. exactly the low-frequency
    band the diagnostic reads as "GW shape is wrong". Same for sRGB-encoded
    (non-linear) JPEG intensities and for uncalibrated light intensities.
"""
import json
import numpy as np
from scipy.ndimage import gaussian_filter

rng = np.random.default_rng(0)
deg = np.degrees


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def ang(a, b):
    return deg(np.arccos(np.clip((unit(a) * unit(b)).sum(-1), -1, 1)))


def masked_blur_scalar(x, mask, s):  # identical to the draft's masked_gaussian_blur
    s = max(s, 0.5)
    return gaussian_filter(x * mask, s) / np.clip(gaussian_filter(mask.astype(float), s), 1e-6, None)


def masked_blur_vec(n, mask, s):
    out = np.stack([masked_blur_scalar(n[..., k], mask, s) for k in range(3)], -1)
    return unit(out)


def rotate_about(n, axis, angle_rad):  # Rodrigues, per pixel
    axis = unit(axis)
    c, s = np.cos(angle_rad)[..., None], np.sin(angle_rad)[..., None]
    return n * c + np.cross(axis, n) * s + axis * (axis * n).sum(-1, keepdims=True) * (1 - c)


# ---------------------------------------------------------------- (A)
H, W = 480, 640
yy, xx = np.mgrid[0:H, 0:W].astype(float)
# smooth "mesh" normal field: gentle drape-like undulation, camera frame (z toward camera = -z_cv)
slope_x = 0.25 * np.sin(2 * np.pi * xx / 400)
slope_y = 0.15 * np.cos(2 * np.pi * yy / 300)
n_mesh = unit(np.stack([-slope_x, -slope_y, -np.ones_like(xx)], -1))
mask = np.ones((H, W), bool)
sigma_px = 6.0   # "one mesh edge" in pixels, as the draft computes it

tangent = unit(np.cross(n_mesh, np.array([0, 1.0, 0])))
results_A = {}
for label, amp_deg in [("weave_ripple_6deg", 6.0), ("weave_ripple_12deg", 12.0)]:
    ripple = np.deg2rad(amp_deg) * np.sin(2 * np.pi * xx / 3.0) * np.sin(2 * np.pi * yy / 3.0)
    ripple = ripple + np.deg2rad(amp_deg / 3) * rng.standard_normal((H, W))   # plus pixel noise
    n_ps = rotate_about(n_mesh, tangent, ripple)
    theta = ang(n_mesh, n_ps)
    draft_low = masked_blur_scalar(theta, mask, sigma_px)
    vec_low = ang(masked_blur_vec(n_mesh, mask, sigma_px), masked_blur_vec(n_ps, mask, sigma_px))
    results_A[label] = {
        "true_low_frequency_disagreement_deg": 0.0,
        "draft_method_rms_low_deg": round(float(np.sqrt((draft_low ** 2).mean())), 2),
        "vector_lowpass_rms_low_deg": round(float(np.sqrt((vec_low ** 2).mean())), 2),
    }

# a REAL low-frequency shape error: smooth 8-deg tilt bump (60 px wide) that the mesh lacks
bump = np.deg2rad(8.0) * np.exp(-(((xx - 320) ** 2 + (yy - 240) ** 2) / (2 * 60.0 ** 2)))
n_ps_bump = rotate_about(n_mesh, tangent, bump)
inside = bump > np.deg2rad(4.0)
draft_low = masked_blur_scalar(ang(n_mesh, n_ps_bump), mask, sigma_px)
vec_low = ang(masked_blur_vec(n_mesh, mask, sigma_px), masked_blur_vec(n_ps_bump, mask, sigma_px))
results_A["real_8deg_bump_no_ripple"] = {
    "draft_method_mean_low_deg_inside_bump": round(float(draft_low[inside].mean()), 2),
    "vector_lowpass_mean_low_deg_inside_bump": round(float(vec_low[inside].mean()), 2),
}

# ---------------------------------------------------------------- (B)
# Camera frame, OpenCV convention. Garment centre 1.5 m in front of the camera.
centre = np.array([0.0, 0.0, 1.5])
# Garment region ~0.5 m wide x 0.9 m tall, +/-0.1 m depth relief
gx, gy, gz = np.meshgrid(np.linspace(-0.25, 0.25, 11), np.linspace(-0.45, 0.45, 19), [-0.1, 0.0, 0.1], indexing="ij")
X = centre + np.stack([gx, gy, gz], -1).reshape(-1, 3)

# 6 lights around the camera axis, 35 deg off-axis, pointing (from the object) back toward the camera side
phis = np.deg2rad(np.arange(0, 360, 60))
off = np.deg2rad(35)
l_dirs = unit(np.stack([np.sin(off) * np.cos(phis), np.sin(off) * np.sin(phis), -np.cos(off) * np.ones_like(phis)], -1))

# surface normals: camera-facing, up to 40 deg from the viewing axis
tilts = np.deg2rad([0, 20, 40]); azs = np.deg2rad(np.arange(0, 360, 45))
N = [np.array([0, 0, -1.0])]
for t in tilts[1:]:
    for a in azs:
        N.append(np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), -np.cos(t)]))
N = np.array(N)


def shade(Xs, n, light_pos=None, light_dir=None, E=None):
    """Lambertian intensities for one normal at many points; point or directional lights."""
    E = np.ones(len(l_dirs)) if E is None else E
    if light_pos is not None:
        v = light_pos[None, :, :] - Xs[:, None, :]              # (P, K, 3)
        r2 = (v ** 2).sum(-1)
        l = v / np.sqrt(r2)[..., None]
        return E[None] * np.clip((l * n).sum(-1), 0, None) / r2   # (P, K)
    return E[None] * np.clip((light_dir[None] * n).sum(-1), 0, None) * np.ones((len(Xs), 1))


def directional_solve(I, L):
    g = np.linalg.lstsq(L, I.T, rcond=None)[0].T
    return unit(g)


results_B = {}
for dL in [1.0, 1.5, 2.0, 3.0]:
    P = centre[None] + dL * l_dirs
    errs = []
    for n in N:
        I = shade(X, n, light_pos=P)
        I = I / (1.0 / dL ** 2)            # scale so centre intensity per unit E == directional model
        n_est = directional_solve(I, l_dirs)
        errs.append(ang(n_est, n[None]))
    errs = np.concatenate(errs)
    results_B[f"near_light_{dL:.1f}m_directional_solve"] = {
        "median_deg": round(float(np.median(errs)), 2),
        "p90_deg": round(float(np.percentile(errs, 90)), 2),
        "max_deg": round(float(errs.max()), 2),
    }

# near-light solve using positions from a mesh that is off by 5 mm (mesh error must not leak into PS)
dL = 1.5; P = centre[None] + dL * l_dirs
errs = []
for n in N:
    I = shade(X, n, light_pos=P)
    Xm = X + 0.005 * unit(rng.standard_normal(X.shape))
    v = P[None] - Xm[:, None]; r2 = (v ** 2).sum(-1); Lp = v / np.sqrt(r2)[..., None]
    for i in range(len(X)):
        g = np.linalg.lstsq(Lp[i] / r2[i][:, None], I[i], rcond=None)[0]
        errs.append(ang(g, n))
errs = np.array(errs)
results_B["near_light_1.5m_pointlight_solve_with_5mm_position_error"] = {
    "median_deg": round(float(np.median(errs)), 3), "max_deg": round(float(errs.max()), 3)}

# sRGB-encoded intensities (JPEG straight into the solver), directional lights, no near-light effect
def srgb_encode(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)

errs = []
for n in N:
    I = shade(X[:1], n, light_dir=l_dirs) * 0.8
    errs.append(ang(directional_solve(srgb_encode(I), l_dirs), n[None]))
errs = np.concatenate(errs)
results_B["srgb_jpeg_values_as_linear"] = {
    "median_deg": round(float(np.median(errs)), 2), "max_deg": round(float(errs.max()), 2)}

# light intensities off by +/-10 % (uncalibrated), directional lights
E = 1 + 0.10 * np.array([1, -1, 0.5, -0.5, 0.8, -0.8])
errs = []
for n in N:
    I = shade(X[:1], n, light_dir=l_dirs, E=E)
    errs.append(ang(directional_solve(I, l_dirs), n[None]))
errs = np.concatenate(errs)
results_B["light_intensities_off_10pct"] = {
    "median_deg": round(float(np.median(errs)), 2), "max_deg": round(float(errs.max()), 2)}


# light positions mis-calibrated by 1 cm (random direction per light), near-light solve, exact surface points
def near_solve(I, Xs, Ppos, E=None):
    E = np.ones(len(Ppos)) if E is None else E
    v = Ppos[None] - Xs[:, None]; r2 = (v ** 2).sum(-1); Lp = v / np.sqrt(r2)[..., None]
    A = Lp * (E[None] / r2)[..., None]                       # (P, K, 3)
    AtA = np.einsum("pki,pkj->pij", A, A); Atb = np.einsum("pki,pk->pi", A, I)
    return unit(np.linalg.solve(AtA, Atb[..., None])[..., 0])

dL = 1.5; P = centre[None] + dL * l_dirs
for err_m in (0.005, 0.010, 0.020):
    P_cal = P + err_m * unit(rng.standard_normal(P.shape))
    errs = np.concatenate([ang(near_solve(shade(X, n, light_pos=P), X, P_cal), n[None]) for n in N])
    results_B[f"light_positions_miscalibrated_{int(err_m*1000)}mm_nearlight_solve"] = {
        "median_deg": round(float(np.median(errs)), 2), "p90_deg": round(float(np.percentile(errs, 90)), 2)}

# 60 cm square softbox (uniform Lambertian emitter, 15x15 elements) facing the garment centre,
# solved as a point light at the softbox centre with a cosine emitter term
def softbox_elements(Pc, size=0.60, n_el=15):
    axis = unit(centre - Pc)                                  # emitter normal, toward the garment
    a = unit(np.cross(axis, [0, 1.0, 0])) if abs(axis[1]) < 0.9 else unit(np.cross(axis, [1.0, 0, 0]))
    b = np.cross(axis, a)
    g = (np.arange(n_el) + 0.5) / n_el - 0.5
    return (Pc + size * (g[:, None, None] * a + g[None, :, None] * b)).reshape(-1, 3), axis

def shade_area(Xs, n, Pc_all):
    out = []
    for Pc in Pc_all:
        el, axis = softbox_elements(Pc)
        v = el[None] - Xs[:, None]; r2 = (v ** 2).sum(-1); l = v / np.sqrt(r2)[..., None]
        emit = np.clip((-l * axis).sum(-1), 0, None)          # Lambertian emitter
        out.append((emit * np.clip((l * n).sum(-1), 0, None) / r2).mean(1))
    return np.stack(out, 1)

def near_solve_cos(I, Xs, Pc_all):
    axes = np.stack([unit(centre - Pc) for Pc in Pc_all])
    v = Pc_all[None] - Xs[:, None]; r2 = (v ** 2).sum(-1); Lp = v / np.sqrt(r2)[..., None]
    emit = np.clip((-Lp * axes[None]).sum(-1), 0, None)
    A = Lp * (emit / r2)[..., None]
    AtA = np.einsum("pki,pkj->pij", A, A); Atb = np.einsum("pki,pk->pi", A, I)
    return unit(np.linalg.solve(AtA, Atb[..., None])[..., 0])

errs = np.concatenate([ang(near_solve_cos(shade_area(X, n, P), X, P), n[None]) for n in N])
results_B["softbox_60cm_at_1.5m_solved_as_point"] = {
    "median_deg": round(float(np.median(errs)), 2), "p90_deg": round(float(np.percentile(errs, 90)), 2)}

print(json.dumps({"A_frequency_separation": results_A, "B_ps_model_bias": results_B}, indent=1))
