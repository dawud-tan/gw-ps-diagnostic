"""Light calibration (config/lights.json) and the near-light irradiance model.

I = rho * E * max(0, -l.a)^mu * max(0, n.l) / r^2, with l the unit vector from the
surface point to the light. For the solve, each light contributes a 3-vector
b = E * max(0, -l.a)^mu / r^2 * l, so that I = (rho n) . b whenever n.l > 0.
An area light is a grid of point elements; its b is the sum over elements, which is
exact while every element is above the surface's horizon.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Light:
    id: int
    role: str
    kind: str
    position: np.ndarray
    E: float = 1.0
    axis: np.ndarray | None = None
    mu: float = 0.0
    size: tuple | None = None

    def to_frame(self, R, t):
        """Apply X' = R X + t (world -> camera) to position and axis."""
        ax = None if self.axis is None else R @ self.axis
        return Light(self.id, self.role, self.kind, R @ self.position + t, self.E, ax, self.mu, self.size)

    def elements(self, n_el=9):
        """(M,3) element positions and per-element E. A point light is one element."""
        if self.kind == "point":
            return self.position[None], np.array([self.E])
        if self.axis is None or self.size is None:
            raise ValueError(f"light {self.id}: an area light needs axis and size_m")
        a = self.axis / np.linalg.norm(self.axis)
        ref = np.array([0, 1.0, 0]) if abs(a[1]) < 0.9 else np.array([1.0, 0, 0])
        e1 = np.cross(a, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(a, e1)
        g = (np.arange(n_el) + 0.5) / n_el - 0.5
        P = (self.position + self.size[0] * g[:, None, None] * e1
             + self.size[1] * g[None, :, None] * e2).reshape(-1, 3)
        return P, np.full(len(P), self.E / len(P))

    def light_vector(self, X):
        """(N,3) points (same frame as the light) -> (N,3) b vectors and (N,) distance."""
        P, E = self.elements()
        b = np.zeros_like(X, dtype=np.float64)
        for Pe, Ee in zip(P, E):
            v = Pe - X
            r = np.linalg.norm(v, axis=1)
            l = v / r[:, None]
            g = Ee / r ** 2
            if self.axis is not None and self.mu != 0:
                a = self.axis / np.linalg.norm(self.axis)
                g = g * np.clip(-(l @ a), 0, None) ** self.mu
            b += g[:, None] * l
        return b, np.linalg.norm(self.position - X, axis=1)


def _parse_light(d):
    return Light(int(d["id"]), d.get("role", "ps"), d.get("kind", "point"),
                 np.asarray(d["position_m"], float), float(d.get("E", 1.0)),
                 None if d.get("axis") is None else np.asarray(d["axis"], float),
                 float(d.get("mu", 0.0)), None if d.get("size_m") is None else tuple(d["size_m"]))


class LightCalibration:
    """Per physical camera: frame ("camera" or "world") and its lights."""

    def __init__(self, cams):
        self.cams = cams

    @classmethod
    def load(cls, path):
        cfg = json.loads(Path(path).read_text())
        cams = {}
        for c in cfg["cameras"]:
            if c["frame"] not in ("camera", "world"):
                raise ValueError(f"camera {c['camera_id']}: frame must be 'camera' or 'world'")
            cams[int(c["camera_id"])] = (c["frame"], [_parse_light(d) for d in c["lights"]])
        return cls(cams)

    def ps_lights_cam(self, camera_id, R, t):
        """PS lights (role 'ps' only) of this camera in the camera frame of a view.

        Camera-frame lights are constant across turntable steps: no per-step rotation.
        """
        frame, lights = self.cams[camera_id]
        ps = [l for l in lights if l.role == "ps"]
        if frame == "world":
            ps = [l.to_frame(np.asarray(R), np.asarray(t)) for l in ps]
        return {l.id: l for l in ps}


def lights_to_json(camera_lights):
    """{camera_id: (frame, [Light])} -> dict in the config/lights.json contract."""
    out = {"cameras": []}
    for cid, (frame, lights) in camera_lights.items():
        out["cameras"].append({"camera_id": cid, "frame": frame, "lights": [
            {"id": l.id, "role": l.role, "kind": l.kind, "position_m": [float(x) for x in l.position],
             "E": float(l.E), "axis": None if l.axis is None else [float(x) for x in l.axis],
             "mu": float(l.mu), "size_m": None if l.size is None else list(l.size)} for l in lights]})
    return out
