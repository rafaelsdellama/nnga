"""Tests for csv dataset loader."""
import pytest
from pathlib import Path
from nnga import get_dataset
from nnga import ROOT
from nnga.configs import cfg

test_directory = Path(ROOT, "tests", "testdata").as_posix()

features_mnist = Path(
    test_directory, "datasets", "classification", "mnist", "features.csv"
).as_posix()
features2_mnist = Path(
    test_directory, "datasets", "classification", "mnist", "features2.csv"
).as_posix()

pytest_output_directory = "./Pytest_output"


@pytest.mark.parametrize(
    "data_directory, is_validation",
    [
        (features_mnist, True),
        (features_mnist, False),
        (features2_mnist, True),
        (features2_mnist, False),
    ],
)
def test_load_dataset_success(data_directory, is_validation):
    """Load csv with success."""
    _cfg = cfg.clone()
    _cfg.MODEL.ARCHITECTURE = "MLP"
    _cfg.OUTPUT_DIR = pytest_output_directory
    _cfg.DATASET.TRAIN_CSV_PATH = data_directory
    _cfg.DATASET.VAL_CSV_PATH = data_directory
    MakeDataset = get_dataset(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    d = MakeDataset(_cfg, True, is_validation)

    assert len(d) > 0
    assert d.n_classes == 10
    assert len(d.labels) > 0 and isinstance(d.labels, list)
    assert len(d.class_weights) > 0 and isinstance(d.class_weights, dict)
    assert len(d.indexes) == 2 and isinstance(d.indexes, tuple)
    assert len(d.indexes[0]) > 0 and isinstance(d.indexes[0], list)
    assert len(d.indexes[1]) > 0 and isinstance(d.indexes[1], list)
    assert isinstance(d.label_encode(d.labels[0]), int)
    assert d.label_decode(d.label_encode(d.labels[0])) == d.labels[0]
    assert d.input_shape > 0 and isinstance(d.input_shape, int)

    assert len(d.scale_parameters) > 0
    assert len(d.load_sample_by_idx(d.indexes[0][0])) > 0
    assert len(d.features) > 0 and isinstance(d.features, list)


@pytest.mark.parametrize(
    "data_directory, is_validation",
    [("wrong_filepath", True), ("wrong_filepath", False)],
)
def test_load_with_wrong_filepath(data_directory, is_validation):
    """Get a wrong path."""
    _cfg = cfg.clone()
    _cfg.MODEL.ARCHITECTURE = "MLP"
    _cfg.DATASET.TRAIN_CSV_PATH = data_directory
    _cfg.DATASET.VAL_CSV_PATH = data_directory
    MakeDataset = get_dataset(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    with pytest.raises(FileNotFoundError):
        MakeDataset(_cfg, True, is_validation)
