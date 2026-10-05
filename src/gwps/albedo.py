"""Albedo for the glTF material: per-channel albedo from the PS normals, a colour correction
fitted to a ColorChecker measured with the same machinery, and sRGB encoding.

Chart: mounted in the ChArUco board's window, so the board pose gives each patch's 3-D
position and normal exactly; each patch's camera-RGB albedo is then the least-squares ratio of
its intensity to the calibrated shading b_k . n over the PS lights (the same model PS uses).
The measured albedos are relative to the primer (stage C folds rho_primer into E); the colour
correction maps them to absolute linear sRGB reflectance, so it also sets the absolute scale.
Colour correction: 3x3 linear (Cheung 2004 degree 1) or root-polynomial degree 2 (Finlayson
2015; exposure-invariant, so albedo scaling is preserved). Validated leave-one-patch-out, ΔE2000.
Any colour transform acts on the final linear albedo only, never on PS inputs.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REFERENCE = "ColorChecker24 - After November 2014"


def reference_linear_srgb(name=REFERENCE):
    """The chart's 24 reference reflectances as linear sRGB (D65, Bradford adaptation)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import colour
    cc = colour.CCS_COLOURCHECKERS[name]
    XYZ = colour.xyY_to_XYZ(np.array(list(cc.data.values())))
    return np.asarray(colour.XYZ_to_RGB(XYZ, colour.RGB_COLOURSPACES["sRGB"], cc.illuminant,
                                        chromatic_adaptation_transform="Bradford")), list(cc.data)


def delta_e00(rgb_a, rgb_b):
    """ΔE2000 between two sets of linear sRGB values."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import colour
    cs = colour.RGB_COLOURSPACES["sRGB"]
    lab = lambda x: colour.XYZ_to_Lab(colour.RGB_to_XYZ(np.clip(x, 1e-6, None), cs, cs.whitepoint),
                                     colour.CCS_ILLUMINANTS["CIE 1931 2 Degree Standard Observer"]["D65"])
    return np.asarray(colour.delta_E(lab(rgb_a), lab(rgb_b), method="CIE 2000"))


def _expand(rgb, method):
    rgb = np.asarray(rgb, float)
    if method == "linear":
        return rgb
    r, g, b = np.clip(rgb, 0, None).T
    return np.c_[rgb, np.sqrt(r * g), np.sqrt(g * b), np.sqrt(r * b)]       # root-polynomial, degree 2


@dataclass
class ColourCorrection:
    method: str
    matrix: np.ndarray            # (3, 3) or (3, 6)

    def apply(self, rgb):
        rgb = np.asarray(rgb, float)
        return (_expand(rgb.reshape(-1, 3), self.method) @ self.matrix.T).reshape(rgb.shape)

    def to_json(self, extra=None):
        return {"method": self.method, "matrix": self.matrix.tolist(), **(extra or {})}

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(d["method"], np.array(d["matrix"]))


def fit_colour_correction(measured, reference, method="linear"):
    """Least squares: reference ≈ M @ expand(measured)."""
    A = _expand(measured, method)
    M, *_ = np.linalg.lstsq(A, np.asarray(reference, float), rcond=None)
    return ColourCorrection(method, M.T)


def leave_one_out_de00(measured, reference, method="linear"):
    errs = []
    for i in range(len(measured)):
        keep = np.arange(len(measured)) != i
        cc = fit_colour_correction(measured[keep], reference[keep], method)
        errs.append(float(delta_e00(cc.apply(measured[i:i + 1]), reference[i:i + 1])[0]))
    return np.array(errs)


def per_channel_albedo(I_rgb, b, n, sat=None, min_shading_frac=0.1):
    """I_rgb: (K,N,3) ambient-subtracted intensities, b: (K,N,3) light vectors, n: (N,3) normals.
    Per channel c: rho_c = sum_k s_k I_kc / sum_k s_k^2 over lights with shading s_k = b_k . n
    above min_shading_frac of the pixel's brightest (attached shadow / grazing excluded)."""
    s = np.einsum("kni,ni->kn", b, n)
    use = s > min_shading_frac * s.max(0, keepdims=True)
    use &= s > 0
    if sat is not None:
        use &= ~sat
    w = np.where(use, s, 0.0)
    den = np.sum(w * s, axis=0)
    rho = np.einsum("kn,knc->nc", w, I_rgb) / np.maximum(den, 1e-12)[:, None]
    return np.where((den > 0)[:, None], rho, 0.0)


def srgb_encode(linear):
    x = np.clip(linear, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def srgb8(linear):
    return np.round(255 * srgb_encode(linear)).astype(np.uint8)


# ---------------------------------------------------------------- chart geometry
@dataclass
class ChartLayout:
    """ColorChecker placement in the board frame (mm): patch (row r, col c) centred at
    origin + (c * pitch, r * pitch); patches sampled over sample_mm squares."""
    origin_mm: tuple
    pitch_mm: float
    sample_mm: float
    rows: int = 4
    cols: int = 6

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))

    def patch_centres_m(self):
        r, c = np.meshgrid(np.arange(self.rows), np.arange(self.cols), indexing="ij")
        x = self.origin_mm[0] + c.ravel() * self.pitch_mm
        y = self.origin_mm[1] + r.ravel() * self.pitch_mm
        return np.stack([x, y], -1) / 1000.0
