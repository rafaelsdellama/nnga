from nnga.utils.data_manipulation import (
    scale_features,
    adjust_image_shape,
    normalize_image,
)

from nnga.utils.data_io import load_image
from nnga import ROOT
from pathlib import Path
import pytest
import numpy as np

test_directory = Path(ROOT, "tests", "testdata").as_posix()

img_test = Path(test_directory, "datasets", "classification", "mnist",
                "0", "3.png").as_posix()

sample = [0, 1]
header = ["Feature_A", "Feature_B"]
scale_parameters = {
    "Feature_A": {"min": 0, "max": 2, "mean": 0, "stdev": 1},
    "Feature_B": {"min": 0, "max": 10, "mean": 5, "stdev": 10},
}


def test_standard_scale():
    scaled = scale_features(sample, header, scale_parameters, "Standard")
    assert scaled == [0.0, -0.4]


def test_minmax_scale():
    scaled = scale_features(sample, header, scale_parameters, "MinMax")
    assert scaled == [0.0, 0.1]


def test_wrong_scale():
    scaled = scale_features(sample, header, scale_parameters, "wrong")
    assert scaled == [0, 1]


@pytest.mark.parametrize(
    "path, input_shape, preserve_ratio",
    [
        (img_test, (100, 100, 1), True),
        (img_test, (100, 100, 1), False),
        (img_test, (300, 150, 1), True),
        (img_test, (150, 300, 1), False),
        (img_test, (100, 100, 3), True),
        (img_test, (100, 100, 3), False),
        (img_test, (300, 150, 3), True),
        (img_test, (150, 300, 3), False),
    ],
)
def test_adjust_image_shape(path, input_shape, preserve_ratio):
    img = adjust_image_shape(load_image(path), input_shape, preserve_ratio)
    assert input_shape == img.shape


@pytest.mark.parametrize(
    "path, input_shape, preserve_ratio",
    [
        (img_test, (100, 100), True),
        (img_test, 300, True),
        (img_test, "(150, 300, 1)", False),
    ],
)
def test_adjust_image_wrong_shape(path, input_shape, preserve_ratio):
    with pytest.raises(ValueError):
        adjust_image_shape(load_image(path), input_shape, preserve_ratio)


@pytest.mark.parametrize(
    "path", [img_test,],
)
def test_normalize_image(path):
    img = normalize_image(load_image(path))
    assert np.amax(img) <= 1.0
