"""Tests for model training."""
import pytest
import os
import shutil
from pathlib import Path
from nnga import ROOT
from nnga.configs import cfg
from nnga import get_dataset, get_architecture
from nnga.utils.logger import setup_logger
from nnga.model_training import ModelTraining

test_directory = Path(ROOT, "tests", "testdata").as_posix()

img_mnist = Path(test_directory, "datasets", "classification",
                 "mnist").as_posix()
img_features = Path(
    test_directory, "datasets", "classification", "mnist", "features.csv"
).as_posix()

img_lung = Path(test_directory, "datasets", "segmentation",
                "lung").as_posix()

pytest_output_directory = "./Pytest_output"
logger = setup_logger("Pytest", pytest_output_directory)


@pytest.mark.parametrize(
    "task, architecture, backbone, img, csv",
    [("Classification", "MLP", "MLP", '', img_features),
     ("Classification", "CNN", "VGG16", img_mnist, ''),
     ("Classification", "CNN/MLP", "VGG16", img_mnist, img_features),
     ("Segmentation", "CNN", "unet", img_lung, '')],
)
def test_fit(task, architecture, backbone, img, csv):
    _cfg = cfg.clone()
    _cfg.OUTPUT_DIR = pytest_output_directory
    _cfg.TASK = task
    _cfg.DATASET.TRAIN_IMG_PATH = img
    _cfg.DATASET.TRAIN_CSV_PATH = csv
    _cfg.DATASET.VAL_IMG_PATH = img
    _cfg.DATASET.VAL_CSV_PATH = csv
    _cfg.MODEL.INPUT_SHAPE = (64, 64, 3)
    _cfg.MODEL.ARCHITECTURE = architecture
    _cfg.MODEL.BACKBONE = backbone
    _cfg.MODEL.FEATURE_SELECTION = False
    _cfg.SOLVER.CROSS_VALIDATION = False
    _cfg.SOLVER.BATCH_SIZE = 16
    _cfg.SOLVER.EPOCHS = 1
    _cfg.SOLVER.OPTIMIZER = "Adam"
    _cfg.SOLVER.BASE_LEARNING_RATE = 0.0001

    MakeDataset = get_dataset(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    datasets = {
        "TRAIN": MakeDataset(_cfg, logger),
        "VAL": MakeDataset(_cfg, logger, is_validation=True),
    }

    if hasattr(datasets["TRAIN"], "scale_parameters"):
        datasets["VAL"].scale_parameters = datasets["TRAIN"].scale_parameters

    MakeModel = get_architecture(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    model = MakeModel(
        _cfg,
        logger,
        datasets["TRAIN"].input_shape,
        datasets["TRAIN"].n_classes,
    )

    # Training
    model_trainner = ModelTraining(_cfg, model, logger, datasets)
    model_trainner.fit()
    shutil.rmtree(pytest_output_directory, ignore_errors=True)


@pytest.mark.parametrize(
    "architecture, backbone",
    [("MLP", "MLP"), ("CNN", "VGG16"), ("CNN/MLP", "VGG16")],
)
def test_train_test_split(architecture, backbone):
    _cfg = cfg.clone()
    _cfg.OUTPUT_DIR = pytest_output_directory
    _cfg.TASK = "Classification"
    _cfg.DATASET.TRAIN_IMG_PATH = img_mnist
    _cfg.DATASET.TRAIN_CSV_PATH = img_features
    _cfg.DATASET.VAL_IMG_PATH = img_mnist
    _cfg.DATASET.VAL_CSV_PATH = img_features
    _cfg.MODEL.INPUT_SHAPE = (64, 64, 3)
    _cfg.MODEL.ARCHITECTURE = architecture
    _cfg.MODEL.BACKBONE = backbone
    _cfg.MODEL.FEATURE_SELECTION = False
    _cfg.SOLVER.CROSS_VALIDATION = False
    _cfg.SOLVER.TEST_SIZE = 0.2
    _cfg.SOLVER.BATCH_SIZE = 16
    _cfg.SOLVER.EPOCHS = 1
    _cfg.SOLVER.OPTIMIZER = "Adam"
    _cfg.SOLVER.BASE_LEARNING_RATE = 0.0001

    MakeDataset = get_dataset(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    datasets = {
        "TRAIN": MakeDataset(_cfg, logger),
        "VAL": MakeDataset(_cfg, logger, is_validation=True),
    }

    if hasattr(datasets["TRAIN"], "scale_parameters"):
        datasets["VAL"].scale_parameters = datasets["TRAIN"].scale_parameters

    MakeModel = get_architecture(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    model = MakeModel(
        _cfg,
        logger,
        datasets["TRAIN"].input_shape,
        datasets["TRAIN"].n_classes,
    )

    # Training
    model_trainner = ModelTraining(_cfg, model, logger, datasets)
    model_trainner.train_test_split()


@pytest.mark.parametrize(
    "architecture, backbone",
    [("MLP", "MLP"), ("CNN", "VGG16"), ("CNN/MLP", "VGG16")],
)
def test_cross_validation(architecture, backbone):
    _cfg = cfg.clone()
    _cfg.OUTPUT_DIR = pytest_output_directory
    _cfg.TASK = "Classification"
    _cfg.DATASET.TRAIN_IMG_PATH = img_mnist
    _cfg.DATASET.TRAIN_CSV_PATH = img_features
    _cfg.DATASET.VAL_IMG_PATH = img_mnist
    _cfg.DATASET.VAL_CSV_PATH = img_features
    _cfg.MODEL.INPUT_SHAPE = (64, 64, 3)
    _cfg.MODEL.ARCHITECTURE = architecture
    _cfg.MODEL.BACKBONE = backbone
    _cfg.MODEL.FEATURE_SELECTION = False
    _cfg.SOLVER.CROSS_VALIDATION = True
    _cfg.SOLVER.CROSS_VALIDATION_FOLDS = 3
    _cfg.SOLVER.TEST_SIZE = 0.2
    _cfg.SOLVER.BATCH_SIZE = 16
    _cfg.SOLVER.EPOCHS = 1
    _cfg.SOLVER.OPTIMIZER = "Adam"
    _cfg.SOLVER.BASE_LEARNING_RATE = 0.0001

    MakeDataset = get_dataset(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    datasets = {
        "TRAIN": MakeDataset(_cfg, logger),
        "VAL": MakeDataset(_cfg, logger, is_validation=True),
    }

    if hasattr(datasets["TRAIN"], "scale_parameters"):
        datasets["VAL"].scale_parameters = datasets["TRAIN"].scale_parameters

    MakeModel = get_architecture(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    model = MakeModel(
        _cfg,
        logger,
        datasets["TRAIN"].input_shape,
        datasets["TRAIN"].n_classes,
    )

    # Training
    model_trainner = ModelTraining(_cfg, model, logger, datasets)
    model_trainner.cross_validation(save=True)
    assert os.path.exists(
        Path(pytest_output_directory, "metrics", "metrics.txt").as_posix()
    )
