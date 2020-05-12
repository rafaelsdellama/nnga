"""Tests for csv dataset loader."""
import pytest
from pathlib import Path

from nnga.datasets.classification.csv_dataset import CSVDataset
from nnga import ROOT
from nnga.configs import cfg

test_directory = Path(ROOT, "tests", "testdata").as_posix()

iris = Path(test_directory, "iris.csv").as_posix()
iris2 = Path(test_directory, "iris2.csv").as_posix()


@pytest.mark.parametrize(
    "data_directory, is_validation",
    [(iris, True), (iris, False), (iris2, True), (iris2, False)],
)
def test_load_dataset_success(data_directory, is_validation):
    """Load csv with success."""
    _cfg = cfg.clone()
    _cfg.DATASET.TRAIN_CSV_PATH = data_directory
    _cfg.DATASET.VAL_CSV_PATH = data_directory
    d = CSVDataset(_cfg, True, is_validation)

    assert len(d) > 0
    assert d.n_classes == 3
    assert len(d.labels) > 0 and type(d.labels) == list
    assert len(d.class_weights) > 0 and type(d.class_weights) == dict
    assert len(d.indexes) > 0 and type(d.indexes) == list
    assert len(d.indexes_labels) > 0 and type(d.indexes_labels) == list
    assert len(d.get_metadata_by_idx(d.indexes[0])) > 0
    assert type(d.label_encode(d.labels[0])) == int
    assert d.label_decode(d.label_encode(d.labels[0])) == d.labels[0]

    assert d.n_features > 0
    assert len(d.scale_parameters) > 0
    assert len(d.load_sample_by_idx(d.indexes[0], "MinMax")) > 0
    assert type(d.features) == list


@pytest.mark.parametrize(
    "data_directory, is_validation",
    [("wrong_filepath", True), ("wrong_filepath", False)],
)
def test_load_with_wrong_filepath(data_directory, is_validation):
    """Get a wrong path."""
    _cfg = cfg.clone()
    _cfg.DATASET.TRAIN_CSV_PATH = data_directory
    _cfg.DATASET.VAL_CSV_PATH = data_directory

    with pytest.raises(FileNotFoundError):
        CSVDataset(_cfg, True, is_validation)
