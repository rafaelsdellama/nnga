"""Tests for image csv dataset loader."""
import pytest
from pathlib import Path

from nnga.datasets.classification.image_csv_dataset import ImageCSVDataset
from nnga import ROOT
from nnga.configs import cfg

test_directory = Path(ROOT, "tests", "testdata").as_posix()

img_mnist = Path(test_directory, "datasets", "mnist").as_posix()
features_mnist = Path(
    test_directory, "datasets", "mnist", "features.csv"
).as_posix()
features2_mnist = Path(
    test_directory, "datasets", "mnist", "features2.csv"
).as_posix()

pytest_output_directory = "./Pytest_output"


@pytest.mark.parametrize(
    "img_dir, data_dir, is_validation, preserve_ratio",
    [
        (img_mnist, features_mnist, True, True),
        (img_mnist, features_mnist, True, False),
        (img_mnist, features_mnist, False, True),
        (img_mnist, features_mnist, False, False),
        (img_mnist, features2_mnist, True, True),
        (img_mnist, features2_mnist, True, False),
        (img_mnist, features2_mnist, False, True),
        (img_mnist, features2_mnist, False, False),
    ],
)
def test_load_dataset_success(
    img_dir, data_dir, is_validation, preserve_ratio
):
    """Load csv and image with success."""
    _cfg = cfg.clone()
    _cfg.OUTPUT_DIR = pytest_output_directory
    _cfg.DATASET.TRAIN_IMG_PATH = img_dir
    _cfg.DATASET.TRAIN_CSV_PATH = data_dir
    _cfg.DATASET.VAL_IMG_PATH = img_dir
    _cfg.DATASET.VAL_CSV_PATH = data_dir
    _cfg.DATASET.PRESERVE_IMG_RATIO = preserve_ratio
    d = ImageCSVDataset(_cfg, True, is_validation)

    assert len(d) > 0
    assert d.n_classes == 10
    assert len(d.labels) > 0 and isinstance(d.labels, list)
    assert len(d.class_weights) > 0 and isinstance(d.class_weights, dict)
    assert len(d.indexes) == 2 and isinstance(d.indexes, tuple)
    assert len(d.indexes[0]) > 0 and isinstance(d.indexes[0], list)
    assert len(d.indexes[1]) > 0 and isinstance(d.indexes[1], list)
    assert isinstance(d.label_encode(d.labels[0]), int)
    assert d.label_decode(d.label_encode(d.labels[0])) == d.labels[0]
    assert len(d.input_shape) == 2 and isinstance(d.input_shape, tuple)
    assert d.input_shape[0] > 0 and isinstance(d.input_shape[0], int)
    assert len(d.input_shape[1]) == 3 and isinstance(d.input_shape[1], tuple)

    assert len(d.scale_parameters) > 0
    assert len(d.load_sample_by_idx(d.indexes[0][0])) > 0
    assert len(d.features) > 0 and isinstance(d.features, list)


@pytest.mark.parametrize(
    "img_dir, data_dir, is_validation",
    [
        ("wrong_filepath", features_mnist, True),
        ("wrong_filepath", features_mnist, False),
        (img_mnist, "wrong_filepath", True),
        (img_mnist, "wrong_filepath", False),
    ],
)
def test_load_with_wrong_filepath(img_dir, data_dir, is_validation):
    _cfg = cfg.clone()
    _cfg.DATASET.TRAIN_IMG_PATH = img_dir
    _cfg.DATASET.TRAIN_CSV_PATH = data_dir
    _cfg.DATASET.VAL_IMG_PATH = img_dir
    _cfg.DATASET.VAL_CSV_PATH = data_dir

    with pytest.raises(FileNotFoundError):
        ImageCSVDataset(_cfg, True, is_validation)
