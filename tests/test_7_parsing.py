"""7. Parsing: an images.txt with an empty POINTS2D line parses correctly; non-pinhole
camera models are rejected."""
import numpy as np
import pytest

from gwps.camera import read_cameras_txt, read_images_txt

IMAGES = """# Image list with two lines of data per image:
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
#   POINTS2D[] as (X, Y, POINT3D_ID)
# Number of images: 3
1 1 0 0 0 0.1 0.2 0.3 1 step0000.png
10.0 20.0 5 30.0 40.0 -1
2 1 0 0 0 0.4 0.5 0.6 1 step0001.png

3 1 0 0 0 0.7 0.8 0.9 1 step0002.png
70 80 5 -1 50 60
"""


def test_empty_points2d_line(tmp_path):
    p = tmp_path / "images.txt"
    p.write_text(IMAGES)
    ims = read_images_txt(p)
    assert sorted(ims) == [1, 2, 3]
    assert [ims[i].name for i in (1, 2, 3)] == ["step0000.png", "step0001.png", "step0002.png"]
    np.testing.assert_allclose(ims[3].tvec, [0.7, 0.8, 0.9])
    assert ims[2].xys.shape == (0, 2)
    assert ims[1].point3D_ids.tolist() == [5, -1]
    assert ims[3].xys.tolist() == [[70, 80], [-1, 50]] and ims[3].point3D_ids.tolist() == [5, 60]


def test_last_image_empty_points_without_trailing_newline(tmp_path):
    p = tmp_path / "images.txt"
    p.write_text("1 1 0 0 0 0 0 0 1 a.png\n1 2 -1\n2 1 0 0 0 0 0 0 1 b.png\n")
    assert [i.name for i in read_images_txt(p).values()] == ["a.png", "b.png"]


def test_non_pinhole_rejected(tmp_path):
    p = tmp_path / "cameras.txt"
    p.write_text("1 OPENCV 640 480 1000 1000 320 240 0.1 0.0 0 0\n")
    with pytest.raises(ValueError, match="PINHOLE"):
        read_cameras_txt(p)
    p.write_text("1 SIMPLE_PINHOLE 640 480 1000 320 240\n2 PINHOLE 640 480 1000 1001 330.7 236.2\n")
    c = read_cameras_txt(p)
    assert (c[1].fx, c[1].fy, c[2].cy) == (1000, 1000, 236.2)
