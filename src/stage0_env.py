"""Stage 0: check the diagnostic environment. Fails loudly if embree is missing."""
import importlib
import sys

REQUIRED = ["numpy", "scipy", "trimesh", "embreex", "cv2", "rawpy", "pycolmap", "PIL"]


def run():
    ok = True
    if sys.version_info < (3, 10):
        print(f"FAIL python {sys.version.split()[0]} < 3.10")
        ok = False
    for m in REQUIRED:
        try:
            mod = importlib.import_module(m)
            print(f"ok   {m} {getattr(mod, '__version__', '')}")
        except Exception as e:
            print(f"FAIL {m}: {e}")
            ok = False
    try:
        import numpy as np
        import trimesh
        from trimesh.ray.ray_pyembree import RayMeshIntersector
        rmi = RayMeshIntersector(trimesh.creation.icosphere(3))
        tri, ray = rmi.intersects_id(np.array([[0, 0, 5.0]]), np.array([[0, 0, -1.0]]), multiple_hits=False)[:2]
        assert len(tri) == 1
        print("ok   embree RayMeshIntersector constructed and casting")
    except Exception as e:
        print(f"FAIL embree ray casting: {e}")
        ok = False
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
