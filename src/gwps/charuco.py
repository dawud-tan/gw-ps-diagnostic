"""ChArUco boards: config, detection in COLMAP pixel convention, pose, intrinsics.

Board frame (OpenCV's): origin at the printed board's top-left outer corner, x along
squares_x, y along squares_y (down the printed image), z = x cross y pointing INTO the
board. The printed face therefore points along -z; in the camera frame its outward
normal is R @ [0, 0, -1].

Pixel convention: OpenCV's cornerSubPix and ArUco marker corners put pixel centres at
integer coordinates, but cv2.aruco.CharucoDetector (checked on 4.11) returns ChArUco
corners +0.5 px off that convention. Every corner is therefore re-refined here with
cornerSubPix and then shifted by +0.5 to COLMAP's convention (pixel centre at +0.5).
Everything downstream of `detect` is COLMAP convention.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)


@dataclass
class BoardConfig:
    dictionary: str
    squares_x: int
    squares_y: int
    square_length_m: float
    marker_length_m: float
    legacy_pattern: bool = False
    fabric_region_m: tuple | None = None      # (x0, y0, x1, y1) in board coordinates
    fabric_thickness_m: float = 0.0

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        if d.get("fabric_region_m") is not None:
            d["fabric_region_m"] = tuple(d["fabric_region_m"])
        return cls(**d)

    def save(self, path):
        Path(path).write_text(json.dumps(self.__dict__, indent=1))

    def board(self):
        d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, self.dictionary))
        b = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_length_m,
                                   self.marker_length_m, d)
        b.setLegacyPattern(self.legacy_pattern)
        return b

    @property
    def size_m(self):
        return self.squares_x * self.square_length_m, self.squares_y * self.square_length_m

    def corners(self):
        """(N,3) chessboard corners in the board frame, indexed by ChArUco corner id."""
        return np.asarray(self.board().getChessboardCorners(), np.float64)

    def usable_corners(self):
        """Corners that are not under or next to the fabric (within half a square of it)."""
        c = self.corners()
        if self.fabric_region_m is None:
            return np.ones(len(c), bool)
        x0, y0, x1, y1 = self.fabric_region_m
        m = 0.5 * self.square_length_m
        inside = (c[:, 0] > x0 - m) & (c[:, 0] < x1 + m) & (c[:, 1] > y0 - m) & (c[:, 1] < y1 + m)
        return ~inside


def to_uint8(img):
    """Linear float image -> 8-bit with a display gamma, for marker detection only."""
    hi = max(float(np.percentile(img, 99.5)), 1e-6)
    return (255 * np.clip(img / hi, 0, 1) ** (1 / 2.2)).astype(np.uint8)


@dataclass
class Detection:
    ids: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    uv: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))   # COLMAP convention
    n_markers: int = 0


def detect(img, cfg: BoardConfig, min_corners=6):
    """Detect ChArUco corners in a linear image. Returns ids and COLMAP-convention (u, v)."""
    board = cfg.board()
    img8 = to_uint8(img)
    cc, ci, mc, mi = cv2.aruco.CharucoDetector(board).detectBoard(img8)
    if ci is None or len(ci) < min_corners:
        return Detection(n_markers=0 if mi is None else len(mi))
    ids = ci.ravel().astype(int)
    keep = cfg.usable_corners()[ids]
    ids, pts = ids[keep], cc.reshape(-1, 2)[keep].astype(np.float32)
    if len(ids) < min_corners:
        return Detection(n_markers=len(mi))
    # window: ~12 % of a square, so it never reaches the marker inside the square
    obj = cfg.corners()[ids]
    d_img = np.linalg.norm(pts[:, None] - pts[None], axis=-1)
    d_obj = np.linalg.norm(obj[:, None, :2] - obj[None, :, :2], axis=-1)
    near = np.isclose(d_obj, cfg.square_length_m, rtol=1e-3)
    sq_px = float(np.median(d_img[near])) if near.any() else 20.0
    win = int(max(3, round(0.12 * sq_px)))
    ref = cv2.cornerSubPix(np.ascontiguousarray(img, dtype=np.float32) / max(float(img.max()), 1e-6),
                           pts.reshape(-1, 1, 2).copy(), (win, win), (-1, -1), SUBPIX_CRITERIA)
    return Detection(ids, ref.reshape(-1, 2).astype(np.float64) + 0.5, len(mi))


def board_pose(det: Detection, cfg: BoardConfig, cam):
    """Board -> camera pose (R, t) from detected corners; cam has rays(uv) and project(X).

    Solved on undistorted normalised coordinates, so PINHOLE and OPENCV cameras are
    handled alike. Returns R, t, and residual statistics in pixels (signed mean per axis
    catches half-pixel convention errors)."""
    obj = cfg.corners()[det.ids]
    r = cam.rays(det.uv)
    xn = (r[:, :2] / r[:, 2:]).astype(np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, xn, np.eye(3), None, flags=cv2.SOLVEPNP_IPPE)
    if not ok:
        raise RuntimeError("solvePnP failed")
    rvec, tvec = cv2.solvePnPRefineLM(obj, xn, np.eye(3), None, rvec, tvec)
    R = cv2.Rodrigues(rvec)[0]
    t = tvec.ravel()
    res = cam.project(obj @ R.T + t) - det.uv
    stats = {"n_corners": int(len(obj)), "rms_px": float(np.sqrt((res ** 2).sum(1).mean())),
             "mean_signed_px": res.mean(0).tolist(), "max_px": float(np.linalg.norm(res, axis=1).max())}
    return R, t, stats


def face_normal_cam(R):
    """Outward normal of the printed face in the camera frame."""
    return R @ np.array([0, 0, -1.0])


def calibrate_intrinsics(detections, cfg: BoardConfig, width, height):
    """OPENCV model (fx, fy, cx, cy, k1, k2, p1, p2) in COLMAP convention from several views."""
    corners = cfg.corners()
    obj = [corners[d.ids].astype(np.float32) for d in detections]
    img = [d.uv.astype(np.float32) for d in detections]
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, (width, height), None, None, flags=cv2.CALIB_FIX_K3)
    k = dist.ravel()
    params = [K[0, 0], K[1, 1], K[0, 2], K[1, 2], k[0], k[1], k[2], k[3]]
    return [float(p) for p in params], float(rms)
