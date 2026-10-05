"""Per-view mesh-vs-PS comparison (camera frame) and per-face accumulation over views.

The two normal fields are low-passed as VECTORS (masked Gaussian, weights = valid x
confidence, then renormalised) and only then compared. The angle map itself is never
low-passed: that rectifies fine texture into fake low-frequency error (evidence A).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

SCALES_MM = (1, 2, 5, 10, 20, 50)


@dataclass
class CompareParams:
    scales_mm: tuple = SCALES_MM
    max_view_angle_deg: float = 75.0
    edge_margin_px: float = 3.0
    depth_jump_rel: float = 0.01       # neighbour depth step > 1 % of depth = discontinuity
    min_conf: float = 0.3
    remove_rotation: bool = True
    min_weight: float = 1e-3           # low-pass support needed, relative to a full kernel
    silhouette_angle_deg: float = 60.0
    taper_px: float = 1.5              # apodisation of the weights at valid-mask edges
    consistency_bin_deg: float = 30.0  # turntable-angle bin; views in one bin count once for consistency


def consistency_bins(images, bin_deg):
    """{image_id: bin} for the consistency statistic. Views are grouped by the azimuth of the
    camera centre about world +Z (the turntable angle), in bins of bin_deg centred on the first
    image's (by name) azimuth, so steps at multiples of the bin width sit mid-bin.

    Why: the mean resultant length is corrected for the number of samples as if their directions
    were independent, but views a few degrees apart see nearly the same light-fixed PS bias. With
    36 x 10 deg steps a camera-frame bias scored 0.711 at s = 50 mm (a false BUILD) against 0.638
    at the 30 deg steps the rule was validated on. Averaging within 30 deg bins restores the
    validated statistic at any finer step. bin_deg <= 0 (or None): one bin per view."""
    ims = sorted(images, key=lambda im: im.name)
    if not ims:
        return {}
    if not bin_deg or bin_deg <= 0:
        return {im.image_id: k for k, im in enumerate(ims)}
    az = {im.image_id: float(np.degrees(np.arctan2(im.centre[1], im.centre[0]))) for im in ims}
    ref = az[ims[0].image_id]
    return {i: int(((a - ref + bin_deg / 2) % 360.0) // bin_deg) for i, a in az.items()}


def unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1)


def angle_deg(a, b):
    """Angle between vectors, in float64 via atan2(|a x b|, a.b). arccos of a float32 dot
    product is quantised near 0 (steps of 0.0198 deg * sqrt(k)), which blurs every angle
    below ~0.2 deg."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1), np.sum(a * b, -1)))


def discontinuity_distance(hit, depth, rel):
    """Distance (px) from each pixel to the nearest hit-mask boundary or depth jump."""
    edge = np.zeros_like(hit)
    for axis in (0, 1):
        a = np.swapaxes(hit, 0, axis)
        z = np.swapaxes(depth, 0, axis)
        e = np.swapaxes(edge, 0, axis)
        jump = (a[1:] != a[:-1]) | (a[1:] & a[:-1] & (np.abs(z[1:] - z[:-1]) > rel * np.maximum(z[1:], z[:-1])))
        e[1:] |= jump
        e[:-1] |= jump
    edge[0, :] = edge[-1, :] = edge[:, 0] = edge[:, -1] = True   # image border
    return distance_transform_edt(~edge)


def kabsch(src, dst, w):
    """Rotation Rk minimising sum w |Rk src - dst|^2; returns Rk and its angle in degrees."""
    Hm = np.einsum("n,ni,nj->ij", w, src, dst)
    U, _, Vt = np.linalg.svd(Hm)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    Rk = Vt.T @ np.diag([1, 1, d]) @ U.T
    return Rk, float(np.degrees(np.arccos(np.clip((np.trace(Rk) - 1) / 2, -1, 1))))


def _gauss(x, sigma):
    k = int(2 * np.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(x, (k, k), sigma, sigmaY=sigma, borderType=cv2.BORDER_CONSTANT)


def _blur(x, sigma):
    """Gaussian blur, zero outside the image. For sigma > 8 px: anti-alias blur (f/2),
    block-average by f ~ sigma/4, blur, bilinear-upsample, with the last blur reduced so the
    variances (f/2)^2 + (f^2-1)/12 (block) + f^2/6 (bilinear) + rest add up to sigma^2.
    The anti-alias step matters: without it, pixel-period texture aliases to DC wherever
    the weights vary inside a block (measured 0.3 deg p99 on the 3 px ripple at s = 20 mm).
    ~10x cheaper than the direct blur at sigma ~ 100 px (s = 50 mm on 1920 px)."""
    if sigma <= 8:
        return _gauss(x, sigma)
    f = int(sigma // 4)
    pre = f / 2
    h, w = x.shape[:2]
    H2, W2 = -(-h // f) * f, -(-w // f) * f
    xp = cv2.copyMakeBorder(_gauss(x, pre), 0, H2 - h, 0, W2 - w, cv2.BORDER_CONSTANT, value=0)
    small = cv2.resize(xp, (W2 // f, H2 // f), interpolation=cv2.INTER_AREA)
    s_small = np.sqrt(sigma ** 2 - pre ** 2 - (f ** 2 - 1) / 12 - f ** 2 / 6) / f
    up = cv2.resize(_gauss(small, s_small), (W2, H2), interpolation=cv2.INTER_LINEAR)
    return up[:h, :w]


def apodise(valid, taper_px):
    """Weights that fall smoothly to 0 at the valid-mask edge. A hard edge truncates pixel-scale
    texture and leaks it into the low band (like a rectangular window in spectral analysis);
    a Gaussian taper of taper_px suppresses a 3 px period by exp(-2 pi^2 taper^2 / 9) ~ 0.007."""
    if taper_px <= 0:
        return valid.astype(np.float32)
    r = int(np.ceil(2 * taper_px))
    er = cv2.erode(valid.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)).astype(np.float32)
    return np.where(valid, _blur(er, taper_px), 0).astype(np.float32)


def lowpass_vec(n, w, sigma_px):
    """Masked Gaussian on each component with weights w, then renormalise. -> (H,W,3), support."""
    num = _blur(np.ascontiguousarray((n * w[..., None]).astype(np.float32)), sigma_px)
    den = _blur(w.astype(np.float32), sigma_px)
    return unit(num / np.maximum(den, 1e-12)[..., None]), den


def valid_mask(mesh_maps, ps_out, garment_faces, p: CompareParams):
    hit = mesh_maps["hit"]
    fid = mesh_maps["face_id"]
    X = mesh_maps["pos_cam"]
    n_mesh = mesh_maps["normal_cam"]
    garment = hit & garment_faces[np.where(hit, fid, 0)]
    cos_view = np.sum(n_mesh * unit(-X), -1)
    dist = discontinuity_distance(hit, mesh_maps["depth"], p.depth_jump_rel)
    valid = (garment & ps_out["ok"] & (ps_out["conf"] >= p.min_conf)
             & (cos_view > np.cos(np.radians(p.max_view_angle_deg))) & (dist >= p.edge_margin_px))
    return valid, cos_view


def compare_view(mesh_maps, ps_out, garment_faces, R, fx, p: CompareParams):
    """-> dict with the valid mask, the per-view rotation angle, and per scale theta (deg) and
    d_world at the valid pixels only (1-D / (N_valid, 3), row-major order of `valid`; theta is
    NaN where the low-pass has no support). cos_view, view_azimuth_deg and elev are also at valid
    pixels. Keeping only valid pixels is what lets a 24 MP view fit in memory.

    elev = Y / Z in the camera frame, i.e. (v - cy) / fy of the pixel's ray (positive down): where
    in the camera's field of view, vertically, the pixel sits. It is what a row-dependent bias
    depends on, and unlike the pixel row it survives stage 1's crop, so board placements (stage 6)
    and garment faces (stage 5) can be compared on it."""
    valid, cos_view = valid_mask(mesh_maps, ps_out, garment_faces, p)
    w = apodise(valid, p.taper_px) * np.where(valid, ps_out["conf"], 0.0)
    n_mesh = mesh_maps["normal_cam"]
    n_ps = ps_out["n"]
    rot_deg = 0.0
    if p.remove_rotation and valid.sum() >= 3:
        Rk, rot_deg = kabsch(n_ps[valid], n_mesh[valid], w[valid])
        n_ps = (n_ps @ Rk.T.astype(np.float32)).astype(np.float32)
    # horizontal angle from the face normal to the direction to the camera, about world +Z:
    # a face's values across turntable steps span the baseline over which consistency is judged
    n_w = n_mesh[valid] @ R
    v_w = unit(-mesh_maps["pos_cam"][valid]) @ R
    az = np.degrees(np.arctan2(n_w[:, 0] * v_w[:, 1] - n_w[:, 1] * v_w[:, 0],
                               n_w[:, 0] * v_w[:, 0] + n_w[:, 1] * v_w[:, 1]))
    Xv = mesh_maps["pos_cam"][valid]
    out = {"valid": valid, "cos_view": cos_view[valid], "rotation_deg": rot_deg, "theta": {}, "d_world": {},
           "sigma_px": {}, "view_azimuth_deg": az, "elev": (Xv[:, 1] / Xv[:, 2]).astype(np.float32)}
    del Xv
    if not valid.any():
        return out
    z_med = float(np.median(mesh_maps["depth"][valid]))
    out["z_med"] = z_med
    for s in p.scales_mm:
        sig = fx * (s / 1000.0) / z_med
        lp_ps, den = lowpass_vec(n_ps, w, sig)
        a = lp_ps[valid]
        support = den[valid] > p.min_weight
        del lp_ps, den
        lp_mesh, _ = lowpass_vec(n_mesh, w, sig)
        b = lp_mesh[valid]
        del lp_mesh
        theta = angle_deg(a, b).astype(np.float32)
        theta[~support] = np.nan
        out["theta"][s] = theta
        out["d_world"][s] = ((a - b) @ R).astype(np.float32)   # R^T (.) per pixel
        out["sigma_px"][s] = sig
    return out


def scatter_valid(values, valid, fill=np.nan):
    """Put per-valid-pixel values back into an (H, W[, C]) image."""
    img = np.full(valid.shape + values.shape[1:], fill, np.float32)
    img[valid] = values
    return img


@dataclass
class FaceAccumulator:
    """Sums over views of per-view per-face means (each view counts once per face).

    Consistency: each view's unit disagreement direction per face is averaged within its
    turntable-angle bin (`add(..., bin_id)`, see consistency_bins), and the mean resultant length
    is taken over bins, so n in its correction counts bins, not views. Views must arrive grouped
    by bin (stage 5 sorts them): only the current bin is held in memory. bin_id=None makes every
    view its own bin.

    Memory (GW meshes have millions of faces): sums are float32 and counts int16 (<= 32767 views),
    and each view works on the faces it sees (np.unique of its face ids), not on n_faces-long
    temporaries per scale. At 10M faces and 24 MP views stage 5 peaked at 9.7 GB before this."""
    n_faces: int
    scales_mm: tuple = SCALES_MM
    theta_sum: dict = field(default_factory=dict)
    dunit_sum: dict = field(default_factory=dict)
    n_views: dict = field(default_factory=dict)
    n_bins: dict = field(default_factory=dict)
    cos_view_sum: np.ndarray = None
    n_views_any: np.ndarray = None

    def __post_init__(self):
        F = self.n_faces
        self._bin_sum, self._bin_seen = {}, {}
        for s in self.scales_mm:
            self.theta_sum[s] = np.zeros(F, np.float32)
            self.dunit_sum[s] = np.zeros((F, 3), np.float32)
            self.n_views[s] = np.zeros(F, np.int16)
            self.n_bins[s] = np.zeros(F, np.int16)
            self._bin_sum[s] = np.zeros((F, 3), np.float32)
            self._bin_seen[s] = np.zeros(F, bool)
        self.cos_view_sum = np.zeros(F, np.float32)
        self.elev_sum = np.zeros(F, np.float32)
        self.n_views_any = np.zeros(F, np.int16)
        self.az_min = np.full(F, np.inf, np.float32)
        self.az_max = np.full(F, -np.inf, np.float32)
        self._bin, self._closed = None, set()

    def _flush(self):
        """Close the current bin: its mean direction per face joins the resultant."""
        for s in self.scales_mm:
            seen = np.flatnonzero(self._bin_seen[s])
            if len(seen):
                self.dunit_sum[s][seen] += unit(self._bin_sum[s][seen].astype(np.float64)).astype(np.float32)
                self.n_bins[s][seen] += 1
                self._bin_sum[s][seen] = 0
                self._bin_seen[s][seen] = False
        if self._bin is not None:
            self._closed.add(self._bin)
        self._bin = None

    def add(self, face_id, cmp, bin_id=None):
        if bin_id is None or bin_id != self._bin:
            self._flush()
            if bin_id is not None:
                if bin_id in self._closed:
                    raise ValueError(f"consistency bin {bin_id} was already closed: add views grouped by bin")
                self._bin = bin_id
        self._add(face_id, cmp)
        if bin_id is None:
            self._flush()

    def _add(self, face_id, cmp):
        valid = cmp["valid"]
        if not valid.any():
            return
        uf, inv = np.unique(face_id[valid], return_inverse=True)     # the faces this view sees
        k = len(uf)
        cnt = np.bincount(inv, minlength=k)
        self.cos_view_sum[uf] += np.bincount(inv, weights=cmp["cos_view"], minlength=k) / cnt
        if "elev" in cmp:
            self.elev_sum[uf] += np.bincount(inv, weights=cmp["elev"], minlength=k) / cnt
        self.n_views_any[uf] += 1
        az = np.bincount(inv, weights=cmp["view_azimuth_deg"], minlength=k) / cnt
        self.az_min[uf] = np.minimum(self.az_min[uf], az)
        self.az_max[uf] = np.maximum(self.az_max[uf], az)
        for s in self.scales_mm:
            th = cmp["theta"][s]
            ok = np.isfinite(th)
            i = inv[ok]
            c = np.bincount(i, minlength=k)
            m = c > 0
            ts = np.bincount(i, weights=th[ok], minlength=k)
            d = cmp["d_world"][s][ok]
            ds = np.stack([np.bincount(i, weights=d[:, j], minlength=k) for j in range(3)], -1)
            f = uf[m]
            self.theta_sum[s][f] += ts[m] / c[m]
            self._bin_sum[s][f] += unit(ds[m]).astype(np.float32)
            self._bin_seen[s][f] = True
            self.n_views[s][f] += 1

    def result(self):
        """mrl_<s> is the raw mean resultant length over bins; n_bins_<s> its sample count;
        dmean_<s> the mean resultant vector (world frame, float32; its norm is mrl_<s>), whose
        direction the verdict checks against the vertical blind spot. elev: the face's mean
        elevation in the camera over the views that saw it (compare_view), which the verdict
        checks against the board placements' (stage 6)."""
        self._flush()
        f32 = np.float32
        out = {"n_views_any": self.n_views_any,
               "elev": np.where(self.n_views_any > 0, self.elev_sum / np.maximum(self.n_views_any, 1),
                                f32(np.nan)).astype(f32),
               "azimuth_span_deg": np.where(self.n_views_any > 0, self.az_max - self.az_min, f32(0)).astype(f32),
               "mean_view_angle_deg": np.degrees(np.arccos(np.clip(
                   self.cos_view_sum / np.maximum(self.n_views_any, 1), -1, 1))).astype(f32)}
        for s in self.scales_mm:
            nv = np.maximum(self.n_views[s], 1).astype(f32)
            nb = np.maximum(self.n_bins[s], 1).astype(f32)
            out[f"theta_{s}"] = np.where(self.n_views[s] > 0, self.theta_sum[s] / nv, f32(np.nan)).astype(f32)
            out[f"mrl_{s}"] = np.where(self.n_bins[s] > 0, np.linalg.norm(self.dunit_sum[s], axis=1) / nb,
                                       f32(np.nan)).astype(f32)
            out[f"n_views_{s}"] = self.n_views[s]
            out[f"n_bins_{s}"] = self.n_bins[s]
            out[f"dmean_{s}"] = (self.dunit_sum[s] / nb[:, None]).astype(f32)
        return out
