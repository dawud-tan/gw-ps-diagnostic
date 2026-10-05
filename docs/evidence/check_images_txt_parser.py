"""Check drafts/build_camera_list.py on a COLMAP images.txt where one image
has no 2D points (its POINTS2D line is empty -- COLMAP writes it that way,
e.g. for models built from known poses before triangulation).

Needs: numpy only.  Run: python check_images_txt_parser.py
"""
import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
DRAFT = HERE.parents[1] / "drafts" / "build_camera_list.py"
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)

IMAGES_TXT = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
1 1 0 0 0 0.1 0.2 0.3 1 step0000.jpg
100.5 200.5 7 300.25 400.75 -1 12.0 34.0 9 56.0 78.0 -1
2 0.7071 0 0.7071 0 0.4 0.5 0.6 1 step0001.jpg

3 0 0 0 1 0.7 0.8 0.9 1 step0002.jpg
10.0 20.0 3 30.0 40.0 -1 50.0 60.0 5 70.0 80.0 11 90.0 95.0 -1
"""
path = OUT / "images.txt"
path.write_text(IMAGES_TXT)


def parse_images_txt_fixed(p):
    """Two lines per image; the POINTS2D line may be empty, so blank lines are
    NOT filtered out. Returns (name, camera_id, t) per image."""
    lines = [l for l in Path(p).read_text().splitlines() if not l.startswith("#")]
    out, i = [], 0
    while i < len(lines):
        if not lines[i].strip():          # stray blank line between records
            i += 1
            continue
        parts = lines[i].split()
        out.append((Path(parts[9]).stem, int(parts[8]), [float(x) for x in parts[5:8]]))
        i += 2                            # skip this image's POINTS2D line (maybe empty)
    return out


spec = importlib.util.spec_from_file_location("bcl", DRAFT)
bcl = importlib.util.module_from_spec(spec); spec.loader.exec_module(bcl)
print("draft parser:")
try:
    for im in bcl.parse_images_txt(path):
        print("  ", im["name"], "camera", im["camera_id"], "t", im["t"])
except Exception as e:
    print("   crashed:", type(e).__name__, e)
print("fixed parser:")
for name, cam_id, t in parse_images_txt_fixed(path):
    print("  ", name, "camera", cam_id, "t", t)
