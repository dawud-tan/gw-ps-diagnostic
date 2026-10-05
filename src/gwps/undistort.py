"""Undistort lit images to the PINHOLE camera the rest of the pipeline uses.

`colmap image_undistorter` only processes registered (SfM) images, so every lit PS image is
undistorted here with the same mapping: from the original distorted camera to the
undistorted PINHOLE camera, bilinear, as COLMAP does. The target is either COLMAP's
undistorted model (garment) or what COLMAP would choose (`colmap_undistorted_camera`, for
the pilot, which has no SfM model).

Pixel convention: both cameras are COLMAP convention (pixel centre at +0.5);
cv2.initUndistortRectifyMap assumes integer pixel centres, so both principal points are
shifted by -0.5 before building the maps. cv2.remap quantises the sampling position to
1/32 px (<= 0.016 px error), the same for every light, so PS ratios are unaffected.

Radiometry: bilinear weights are positive, so linear intensities stay linear (no ringing
from cubic/Lanczos near shadows or clipping). A pixel that draws on any saturated source
pixel is written as full scale (1.0), so load_linear still flags it as saturated.
"""
from __future__ import annotations

import cv2
import numpy as np

from .camera import Camera, OpenCVCamera


def colmap_undistorted_camera(src: OpenCVCamera, camera_id=None):
    """The PINHOLE camera COLMAP's image_undistorter would pick (default options)."""
    import pycolmap
    u = pycolmap.undistort_camera(pycolmap.UndistortCameraOptions(), src.to_pycolmap())
    return Camera.from_colmap(camera_id or src.camera_id, u.model_name, u.width, u.height, list(u.params))


def undistort_maps(src: OpenCVCamera, dst: Camera):
    """float32 remap maps (dst pixel index -> src pixel index, OpenCV convention)."""
    Ks = src.K.copy()
    Ks[:2, 2] -= 0.5
    Kd = dst.K.copy()
    Kd[:2, 2] -= 0.5
    return cv2.initUndistortRectifyMap(Ks, src.dist, None, Kd, (dst.width, dst.height), cv2.CV_32FC1)


def undistort_image(img, maps, sat=None):
    """Bilinear remap of a linear image; saturation propagated as full scale."""
    m1, m2 = maps
    out = cv2.remap(np.ascontiguousarray(img, dtype=np.float32), m1, m2, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if sat is not None and sat.any():
        s = cv2.remap(sat.astype(np.float32), m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        out[s > 0] = 1.0
    return out


def source_coverage(maps, src_shape):
    """True where the bilinear support lies entirely inside the source image."""
    m1, m2 = maps
    h, w = src_shape
    return (m1 >= 0) & (m1 <= w - 1) & (m2 >= 0) & (m2 <= h - 1)
