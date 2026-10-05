"""C2. Intrinsics from ChArUco shots through a distorted OPENCV camera. The principal point
trades off against a small global rotation of the camera frame (harmless: lights, poses and
normals all live in the same frame), so the test is on the residual ray error after
removing the best global rotation."""
import numpy as np

from gwps.camera import OpenCVCamera
from gwps.compare import angle_deg, kabsch


def test_intrinsics_ray_error(pilot):
    d = pilot.data()
    cam, info = pilot.intrinsics(d)
    true = OpenCVCamera(1, cam.width, cam.height, pilot.truth(d)["distorted_camera_params"])
    j, i = np.meshgrid(np.linspace(0.5, cam.width - 0.5, 97), np.linspace(0.5, cam.height - 0.5, 73))
    uv = np.stack([j.ravel(), i.ravel()], -1)
    re, rt = cam.rays(uv), true.rays(uv)
    Rk, rot = kabsch(re, rt, np.ones(len(uv)))
    px = np.radians(angle_deg(re @ Rk.T, rt)) * true.fx
    c = info["corner_coverage_u_min_max_v_min_max"]
    inside = (uv[:, 0] > c[0] * cam.width) & (uv[:, 0] < c[1] * cam.width) & \
             (uv[:, 1] > c[2] * cam.height) & (uv[:, 1] < c[3] * cam.height)
    print(f"\n[C2] rms {info['rms_px']:.3f} px over {info['n_views']} views; principal point error "
          f"{np.round(np.array(cam.params[2:4]) - true.params[2:4], 2)} px ~ global rotation {rot:.4f} deg; "
          f"residual ray error median {np.median(px):.3f}, max inside corner coverage {px[inside].max():.3f}, "
          f"max anywhere {px.max():.3f} px")
    assert rot < 0.1
    # 0.5 px at f = 3000 is 0.01 deg of ray direction: harmless for poses, ball centres and PS
    assert px[inside].max() < 0.5
