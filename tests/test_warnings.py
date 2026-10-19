"""Quality-gate warnings, input contract, and asymmetry ordering."""
import numpy as np
import cv2
import pytest

from skinlens.pipeline import analyze, asymmetry, resize_rgb


def test_tiny_lesion_warns_and_is_unreliable():
    im = np.full((256, 256, 3), (215, 175, 150), np.uint8)
    cv2.circle(im, (128, 128), 8, (85, 45, 30), -1)
    res = analyze(im)
    assert any("Very small" in w for w in res["features"].warnings)
    assert res["features"].band == "Unreliable"


def test_oversized_region_warns():
    # Dark field with one bright spot: Otsu selects the dark interior,
    # which fills nearly the whole dermatoscope ellipse.
    im = np.full((256, 256, 3), (70, 50, 40), np.uint8)
    cv2.circle(im, (90, 90), 25, (225, 205, 185), -1)
    res = analyze(im)
    assert res["features"].area_fraction > 0.75
    assert any("Most of the image" in w for w in res["features"].warnings)
    assert res["features"].band == "Unreliable"


def test_resize_input_contract():
    with pytest.raises(ValueError):
        resize_rgb(np.zeros((64, 64, 3), np.float32))  # wrong dtype
    with pytest.raises(ValueError):
        resize_rgb(np.zeros((31, 31, 3), np.uint8))  # below the 32px floor
    assert resize_rgb(np.zeros((32, 32, 3), np.uint8)).shape == (32, 32, 3)


def test_resize_downscales_to_max_side_and_preserves_ratio():
    out = resize_rgb(np.zeros((500, 1000, 3), np.uint8))
    assert out.shape == (256, 512, 3)  # MAX_SIDE = 512
    assert resize_rgb(np.zeros((200, 300, 3), np.uint8)).shape == (200, 300, 3)  # small enough: untouched


def test_asymmetry_orders_lopsided_above_circle():
    yy, xx = np.mgrid[:200, :200]
    circle = (xx - 100) ** 2 + (yy - 100) ** 2 <= 40 ** 2
    lopsided = (((xx - 100) / 70) ** 2 + ((yy - 100) / 30) ** 2 <= 1) & ~((xx - 140) ** 2 + (yy - 100) ** 2 <= 25 ** 2)
    assert asymmetry(circle) < asymmetry(lopsided)
    assert asymmetry(np.zeros((10, 10), bool)) == 0.0  # too few points to judge
