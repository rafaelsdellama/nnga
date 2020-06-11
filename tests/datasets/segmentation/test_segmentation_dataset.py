"""Tests for segmentation dataset loader."""
import pytest
from pathlib import Path
from nnga import get_dataset
from nnga import ROOT
from nnga.configs import cfg

test_directory = Path(ROOT, "tests", "testdata").as_posix()
img_lung = Path(test_directory, "datasets", "segmentation",
                "lung").as_posix()
pytest_output_directory = "./Pytest_output"


@pytest.mark.parametrize(
    "data_directory, is_validation, preserve_ratio",
    [
        (img_lung, True, True),
        (img_lung, True, False),
        (img_lung, False, True),
        (img_lung, False, False),
    ],
)
def test_load_dataset_success(data_directory, is_validation, preserve_ratio):
    """Load image with success."""
    _cfg = cfg.clone()
    _cfg.TASK = "Segmentation"
    _cfg.MODEL.ARCHITECTURE = "CNN"
    _cfg.OUTPUT_DIR = pytest_output_directory
    _cfg.DATASET.TRAIN_IMG_PATH = data_directory
    _cfg.DATASET.VAL_IMG_PATH = data_directory
    _cfg.DATASET.PRESERVE_IMG_RATIO = preserve_ratio
    MakeDataset = get_dataset(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    d = MakeDataset(_cfg, True, is_validation)

    assert len(d) > 0
    assert d.n_classes == 2
    assert len(d.input_shape) == 3 and isinstance(d.input_shape, tuple)


@pytest.mark.parametrize(
    "data_directory, is_validation",
    [("wrong_filepath", True), ("wrong_filepath", False)],
)
def test_load_with_wrong_filepath(data_directory, is_validation):
    """Get a wrong path."""
    _cfg = cfg.clone()
    _cfg.TASK = "Segmentation"
    _cfg.MODEL.ARCHITECTURE = "CNN"
    _cfg.DATASET.TRAIN_IMG_PATH = data_directory
    _cfg.DATASET.VAL_IMG_PATH = data_directory
    MakeDataset = get_dataset(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)

    with pytest.raises(FileNotFoundError):
        MakeDataset(_cfg, True, is_validation)


@pytest.mark.parametrize(
    "data_directory, preserve_ratio", [(img_lung, "dsadsa"), (img_lung, 2),],
)
def test_wrong_preserve_ratio(data_directory, preserve_ratio):
    _cfg = cfg.clone()
    _cfg.TASK = "Segmentation"
    _cfg.MODEL.ARCHITECTURE = "CNN"
    _cfg.DATASET.TRAIN_IMG_PATH = data_directory
    _cfg.DATASET.VAL_IMG_PATH = data_directory
    _cfg.DATASET.PRESERVE_IMG_RATIO = preserve_ratio
    MakeDataset = get_dataset(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)

    with pytest.raises(ValueError):
        MakeDataset(_cfg, True)


@pytest.mark.parametrize(
    "data_directory, input_shape", [(img_lung, (10, 10)), (img_lung, 10),],
)
def test_wrong_input_shape(data_directory, input_shape):
    _cfg = cfg.clone()
    _cfg.TASK = "Segmentation"
    _cfg.MODEL.ARCHITECTURE = "CNN"
    _cfg.DATASET.TRAIN_IMG_PATH = data_directory
    _cfg.DATASET.VAL_IMG_PATH = data_directory
    _cfg.MODEL.INPUT_SHAPE = input_shape
    MakeDataset = get_dataset(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)

    with pytest.raises(ValueError):
        MakeDataset(_cfg, True)


@pytest.mark.parametrize(
    "data_directory, shuffe", [(img_lung, "wrong"), (img_lung, 10),],
)
def test_wrong_shuffe(data_directory, shuffe):
    _cfg = cfg.clone()
    _cfg.TASK = "Segmentation"
    _cfg.MODEL.ARCHITECTURE = "CNN"
    _cfg.DATASET.TRAIN_IMG_PATH = data_directory
    _cfg.DATASET.TRAIN_SHUFFLE = shuffe
    _cfg.DATASET.VAL_IMG_PATH = data_directory
    _cfg.DATASET.VAL_SHUFFLE = shuffe
    MakeDataset = get_dataset(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)

    with pytest.raises(ValueError):
        MakeDataset(_cfg, True)


@pytest.mark.parametrize(
    "data_directory, augmentation", [(img_lung, "wrong"), (img_lung, 10),],
)
def test_wrong_augmentation(data_directory, augmentation):
    _cfg = cfg.clone()
    _cfg.TASK = "Segmentation"
    _cfg.MODEL.ARCHITECTURE = "CNN"
    _cfg.DATASET.TRAIN_IMG_PATH = data_directory
    _cfg.DATASET.TRAIN_AUGMENTATION = augmentation
    _cfg.DATASET.VAL_IMG_PATH = data_directory
    _cfg.DATASET.VAL_AUGMENTATION = augmentation
    MakeDataset = get_dataset(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)

    with pytest.raises(ValueError):
        MakeDataset(_cfg, True)
