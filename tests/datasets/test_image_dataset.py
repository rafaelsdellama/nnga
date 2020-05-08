"""Tests for image dataset loader."""
import pytest
from pathlib import Path

from nnga.datasets.image_dataset import ImageDataset
from nnga import ROOT
from nnga.configs import cfg

test_directory = Path(ROOT, "tests", "testdata").as_posix()

img_directory = Path(test_directory, "img_dataset").as_posix()


@pytest.mark.parametrize(
    "data_directory, is_validation, preserve_ratio",
    [
        (img_directory, True, True),
        (img_directory, True, False),
        (img_directory, False, True),
        (img_directory, False, False),
    ],
)
def test_load_dataset_success(data_directory, is_validation, preserve_ratio):
    """Load csv with success."""
    _cfg = cfg.clone()
    _cfg.DATASET.TRAIN_IMG_PATH = data_directory
    _cfg.DATASET.VAL_IMG_PATH = data_directory
    _cfg.DATASET.PRESERVE_IMG_RATIO = preserve_ratio
    d = ImageDataset(_cfg, True, is_validation)

    assert len(d) > 0
    assert d.n_classes == 2
    assert len(d.labels) > 0 and type(d.labels) == list
    assert len(d.class_weights) > 0 and type(d.class_weights) == dict
    assert len(d.indexes) > 0 and type(d.indexes) == list
    assert len(d.indexes_labels) > 0 and type(d.indexes_labels) == list
    assert len(d.get_metadata_by_idx(d.indexes[0])) > 0
    assert type(d.label_encode(d.labels[0])) == int
    assert d.label_decode(d.label_encode(d.labels[0])) == d.labels[0]


@pytest.mark.parametrize(
    "data_directory, is_validation",
    [("wrong_filepath", True), ("wrong_filepath", False)],
)
def test_load_with_wrong_filepath(data_directory, is_validation):
    """Get a wrong path."""
    _cfg = cfg.clone()
    _cfg.DATASET.TRAIN_IMG_PATH = data_directory
    _cfg.DATASET.VAL_IMG_PATH = data_directory

    with pytest.raises(FileNotFoundError):
        ImageDataset(_cfg, True, is_validation)


@pytest.mark.parametrize(
    "data_directory, preserve_ratio",
    [(img_directory, "dsadsa"), (img_directory, 2),],
)
def test_wrong_preserve_ratio(data_directory, preserve_ratio):
    _cfg = cfg.clone()
    _cfg.DATASET.TRAIN_IMG_PATH = data_directory
    _cfg.DATASET.VAL_IMG_PATH = data_directory
    _cfg.DATASET.PRESERVE_IMG_RATIO = preserve_ratio

    with pytest.raises(ValueError):
        ImageDataset(_cfg, True)


@pytest.mark.parametrize(
    "data_directory, input_shape",
    [(img_directory, (10, 10)), (img_directory, 10),],
)
def test_wrong_input_shape(data_directory, input_shape):
    _cfg = cfg.clone()
    _cfg.DATASET.TRAIN_IMG_PATH = data_directory
    _cfg.DATASET.VAL_IMG_PATH = data_directory
    _cfg.MODEL.INPUT_SHAPE = input_shape

    with pytest.raises(ValueError):
        ImageDataset(_cfg, True)


@pytest.mark.parametrize(
    "data_directory, shuffe", [(img_directory, "wrong"), (img_directory, 10),],
)
def test_wrong_shuffe(data_directory, shuffe):
    _cfg = cfg.clone()
    _cfg.DATASET.TRAIN_IMG_PATH = data_directory
    _cfg.DATASET.TRAIN_SHUFFLE = shuffe
    _cfg.DATASET.VAL_IMG_PATH = data_directory
    _cfg.DATASET.VAL_SHUFFLE = shuffe

    with pytest.raises(ValueError):
        ImageDataset(_cfg, True)


@pytest.mark.parametrize(
    "data_directory, augmentation",
    [(img_directory, "wrong"), (img_directory, 10),],
)
def test_wrong_augmentation(data_directory, augmentation):
    _cfg = cfg.clone()
    _cfg.DATASET.TRAIN_IMG_PATH = data_directory
    _cfg.DATASET.TRAIN_AUGMENTATION = augmentation
    _cfg.DATASET.VAL_IMG_PATH = data_directory
    _cfg.DATASET.VAL_AUGMENTATION = augmentation

    with pytest.raises(ValueError):
        ImageDataset(_cfg, True)
