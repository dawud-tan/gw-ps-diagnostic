"""Pinhole cameras, COLMAP text I/O and projection, in COLMAP/OpenCV convention.

Camera frame: x right, y down, z forward. X_cam = R @ X_world + t, R from the
images.txt quaternion (world -> camera). The centre of pixel (row i, col j) is at
(u, v) = (j + 0.5, i + 0.5). Only PINHOLE and SIMPLE_PINHOLE are accepted: everything
downstream works on the undistorted model from `colmap image_undistorter`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PINHOLE_MODELS = ("PINHOLE", "SIMPLE_PINHOLE")


@dataclass
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_colmap(cls, camera_id, model, width, height, params):
        if model not in PINHOLE_MODELS:
            raise ValueError(
                f"camera {camera_id}: model {model} is not PINHOLE/SIMPLE_PINHOLE; "
                "run colmap image_undistorter and use its model")
        p = [float(x) for x in params]
        if model == "SIMPLE_PINHOLE":
            f, cx, cy = p
            return cls(camera_id, model, int(width), int(height), f, f, cx, cy)
        fx, fy, cx, cy = p
        return cls(camera_id, model, int(width), int(height), fx, fy, cx, cy)

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def project(self, X_cam):
        """(N,3) camera-frame points -> (N,2) pixel coordinates (u, v)."""
        X = np.asarray(X_cam, float)
        z = X[:, 2]
        return np.stack([self.fx * X[:, 0] / z + self.cx, self.fy * X[:, 1] / z + self.cy], -1)

    def rays(self, uv):
        """(N,2) pixel coordinates -> (N,3) unit ray directions in the camera frame."""
        uv = np.asarray(uv, float)
        d = np.stack([(uv[:, 0] - self.cx) / self.fx, (uv[:, 1] - self.cy) / self.fy,
                      np.ones(len(uv))], -1)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def pixel_centres(self):
        """(H*W,2) (u, v) of every pixel centre, row-major, so index = i * W + j."""
        j, i = np.meshgrid(np.arange(self.width), np.arange(self.height))
        return np.stack([j.ravel() + 0.5, i.ravel() + 0.5], -1)

    def colmap_params(self):
        if self.model == "SIMPLE_PINHOLE":
            return [self.fx, self.cx, self.cy]
        return [self.fx, self.fy, self.cx, self.cy]


@dataclass
class Image:
    image_id: int
    qvec: np.ndarray          # (w, x, y, z), world -> camera
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))

    @property
    def R(self):
        return qvec_to_R(self.qvec)

    @property
    def t(self):
        return np.asarray(self.tvec, float)

    @property
    def centre(self):
        return -self.R.T @ self.t

    @property
    def stem(self):
        return Path(self.name).stem

    def world_to_cam(self, X):
        return np.asarray(X, float) @ self.R.T + self.t


def qvec_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_qvec(R):
    R = np.asarray(R, float)
    tr = np.trace(R)
    if tr > 0:
        s = 2 * np.sqrt(tr + 1)
        q = [s / 4, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        k = int(np.argmax(np.diag(R)))
        i, j = (k + 1) % 3, (k + 2) % 3
        s = 2 * np.sqrt(1 + R[k, k] - R[i, i] - R[j, j])
        q = np.zeros(4)
        q[0] = (R[j, i] - R[i, j]) / s
        q[1 + k] = s / 4
        q[1 + i] = (R[i, k] + R[k, i]) / s
        q[1 + j] = (R[j, k] + R[k, j]) / s
    q = np.asarray(q)
    return q if q[0] >= 0 else -q


def _data_lines(path):
    return [ln.rstrip("\r\n") for ln in Path(path).read_text().splitlines() if not ln.startswith("#")]


def read_cameras_txt(path):
    cams = {}
    for ln in _data_lines(path):
        if not ln.strip():
            continue
        tok = ln.split()
        cid = int(tok[0])
        cams[cid] = Camera.from_colmap(cid, tok[1], tok[2], tok[3], tok[4:])
    return cams


def read_images_txt(path):
    """Two lines per image; the POINTS2D line may be empty and must not be dropped."""
    lines = _data_lines(path)
    images = {}
    i = 0
    while i < len(lines):
        head = lines[i].split()
        if not head:              # only possible as trailing whitespace between records
            i += 1
            continue
        if len(head) < 10:
            raise ValueError(f"{path}: malformed image line {i}: {lines[i]!r}")
        pts = lines[i + 1].split() if i + 1 < len(lines) else []
        i += 2
        arr = np.asarray(pts, float).reshape(-1, 3) if pts else np.zeros((0, 3))
        img = Image(int(head[0]), np.array(head[1:5], float), np.array(head[5:8], float),
                    int(head[8]), " ".join(head[9:]), arr[:, :2], arr[:, 2].astype(np.int64))
        images[img.image_id] = img
    return images


def read_points3D_txt(path):
    pts = {}
    if not Path(path).exists():
        return pts
    for ln in _data_lines(path):
        tok = ln.split()
        if tok:
            pts[int(tok[0])] = np.array(tok[1:4], float)
    return pts


def write_colmap_txt(sparse_dir, cameras, images, points3D=None):
    d = Path(sparse_dir)
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "cameras.txt", "w") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for c in cameras.values():
            f.write(f"{c.camera_id} {c.model} {c.width} {c.height} "
                    + " ".join(repr(float(p)) for p in c.colmap_params()) + "\n")
    with open(d / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for im in images.values():
            f.write(f"{im.image_id} " + " ".join(repr(float(x)) for x in im.qvec) + " "
                    + " ".join(repr(float(x)) for x in im.tvec) + f" {im.camera_id} {im.name}\n")
            f.write(" ".join(f"{x:.4f} {y:.4f} {int(p)}" for (x, y), p in zip(im.xys, im.point3D_ids)) + "\n")
    with open(d / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        for pid, X in (points3D or {}).items():
            X, err = (X if isinstance(X, tuple) else (X, 0.0))
            f.write(f"{pid} {X[0]!r} {X[1]!r} {X[2]!r} 128 128 128 {err!r}\n")


def load_model(sparse_dir):
    d = Path(sparse_dir)
    return read_cameras_txt(d / "cameras.txt"), read_images_txt(d / "images.txt")


# COLMAP perspective models that map onto OpenCV's distortion vector (k1, k2, p1, p2[, k3..k6]).
_DISTORTED = {
    "SIMPLE_PINHOLE": lambda p: ([p[0], p[0], p[1], p[2]], []),
    "PINHOLE": lambda p: (list(p[:4]), []),
    "SIMPLE_RADIAL": lambda p: ([p[0], p[0], p[1], p[2]], [p[3], 0, 0, 0]),
    "RADIAL": lambda p: ([p[0], p[0], p[1], p[2]], [p[3], p[4], 0, 0]),
    "OPENCV": lambda p: (list(p[:4]), list(p[4:8])),
    "FULL_OPENCV": lambda p: (list(p[:4]), list(p[4:12])),   # k1 k2 p1 p2 k3 k4 k5 k6
}


@dataclass
class OpenCVCamera:
    """A distorted COLMAP camera (SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV; or a pinhole),
    in COLMAP pixel convention. `params` is always [fx, fy, cx, cy, k1, k2, p1, p2(, k3..k6)];
    `colmap_model` / `colmap_params` keep the original so COLMAP's own choices can be reproduced.

    Used by calibration (stage C, which undistorts points, not pixels) and by lit-image
    undistortion. Everything downstream of undistortion requires PINHOLE/SIMPLE_PINHOLE."""
    camera_id: int
    width: int
    height: int
    params: list
    colmap_model: str = "OPENCV"
    colmap_params: list = None

    model = "OPENCV"

    def __post_init__(self):
        if self.colmap_params is None:
            self.colmap_params = list(self.params)

    @classmethod
    def from_colmap(cls, camera_id, model, width, height, params):
        if model not in _DISTORTED:
            raise ValueError(f"camera {camera_id}: model {model} is not supported (fisheye models need cv2.fisheye)")
        k, d = _DISTORTED[model]([float(x) for x in params])
        return cls(int(camera_id), int(width), int(height), k + (d or [0.0, 0.0, 0.0, 0.0]), model,
                   [float(x) for x in params])

    @property
    def K(self):
        fx, fy, cx, cy = self.params[:4]
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])

    @property
    def dist(self):
        d = list(self.params[4:])
        return np.array(d if len(d) == 8 else d + [0.0])      # OpenCV: k1 k2 p1 p2 k3 [k4 k5 k6]

    fx = property(lambda self: self.params[0])
    fy = property(lambda self: self.params[1])
    cx = property(lambda self: self.params[2])
    cy = property(lambda self: self.params[3])

    def rays(self, uv):
        import cv2
        uv = np.asarray(uv, np.float64).reshape(-1, 1, 2)
        xn = cv2.undistortPointsIter(uv, self.K, self.dist, None, None,
                                     (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-12)).reshape(-1, 2)
        d = np.concatenate([xn, np.ones((len(xn), 1))], 1)
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def project(self, X_cam):
        import cv2
        X = np.asarray(X_cam, np.float64).reshape(-1, 1, 3)
        uv, _ = cv2.projectPoints(X, np.zeros(3), np.zeros(3), self.K, self.dist)
        return uv.reshape(-1, 2)

    def pixel_centres(self):
        j, i = np.meshgrid(np.arange(self.width), np.arange(self.height))
        return np.stack([j.ravel() + 0.5, i.ravel() + 0.5], -1)

    def to_pycolmap(self):
        import pycolmap
        return pycolmap.Camera(model=self.colmap_model, width=self.width, height=self.height,
                               params=self.colmap_params)

    def to_json(self, extra=None):
        d = {"camera_id": self.camera_id, "model": "OPENCV", "width": self.width, "height": self.height,
             "params": [float(p) for p in self.params[:8]],
             "_convention": "COLMAP: centre of pixel (row i, col j) is (j + 0.5, i + 0.5)"}
        d.update(extra or {})
        return d


def read_distorted_cameras_txt(path):
    """All cameras of a cameras.txt (any supported perspective model) as OpenCVCamera."""
    cams = {}
    for ln in _data_lines(path):
        tok = ln.split()
        if tok:
            cams[int(tok[0])] = OpenCVCamera.from_colmap(int(tok[0]), tok[1], tok[2], tok[3], tok[4:])
    return cams


def load_camera_model(path, camera_id=None):
    """A PINHOLE Camera from cameras.txt, or an OpenCVCamera from a stage-C intrinsics JSON."""
    import json
    p = Path(path)
    if p.suffix == ".json":
        d = json.loads(p.read_text())
        if d["model"] != "OPENCV":
            raise ValueError(f"{p}: expected an OPENCV intrinsics file")
        return OpenCVCamera.from_colmap(int(d["camera_id"]), "OPENCV", d["width"], d["height"], d["params"])
    cams = read_cameras_txt(p)
    return cams[camera_id] if camera_id is not None else next(iter(cams.values()))


def read_points3D_txt_full(path):
    """{point3D_id: (xyz, reprojection error in px)} from a COLMAP points3D.txt."""
    pts = {}
    if not Path(path).exists():
        return pts
    for ln in _data_lines(path):
        tok = ln.split()
        if tok:
            pts[int(tok[0])] = (np.array(tok[1:4], float), float(tok[7]) if len(tok) > 7 else 0.0)
    return pts


def load_colmap_dir(path, workdir=None):
    """Cameras, images and points (with errors) of a COLMAP model in text or binary form.
    Binary models are converted to text with pycolmap first."""
    p = Path(path)
    if not (p / "cameras.txt").exists() and (p / "cameras.bin").exists():
        import pycolmap
        import tempfile
        workdir = Path(workdir or tempfile.mkdtemp(prefix="colmap_txt_"))
        pycolmap.Reconstruction(str(p)).write_text(str(workdir))
        p = workdir
    return read_cameras_txt(p / "cameras.txt"), read_images_txt(p / "images.txt"), read_points3D_txt_full(p / "points3D.txt")
