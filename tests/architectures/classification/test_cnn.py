import pytest
from pathlib import Path
import os
from nnga.configs import cfg
from nnga import get_architecture
from nnga.utils.logger import setup_logger


pytest_output_directory = "./Pytest_output"
logger = setup_logger("Pytest", pytest_output_directory)


@pytest.mark.parametrize(
    "architecture, backbone, input_shape, n_classes",
    [("CNN", "VGG16", (200, 200, 3), 2),],
)
def test_create_model(architecture, backbone, input_shape, n_classes):
    _cfg = cfg.clone()
    _cfg.MODEL.ARCHITECTURE = architecture
    _cfg.MODEL.BACKBONE = backbone
    _cfg.MODEL.FEATURE_SELECTION = False
    MakeModel = get_architecture(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    MakeModel(_cfg, logger, input_shape, n_classes)


@pytest.mark.parametrize(
    "architecture, backbone, input_shape, n_classes",
    [("CNN", "wrong", (200, 200, 3), 2),],
)
def test_create_model_wrong_backbone(
    architecture, backbone, input_shape, n_classes
):
    _cfg = cfg.clone()
    _cfg.MODEL.ARCHITECTURE = architecture
    _cfg.MODEL.BACKBONE = backbone
    _cfg.MODEL.FEATURE_SELECTION = False
    MakeModel = get_architecture(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    with pytest.raises(RuntimeError):
        MakeModel(_cfg, logger, input_shape, n_classes)


@pytest.mark.parametrize(
    "architecture, backbone, input_shape, n_classes",
    [("CNN", "VGG16", (200, 200, 3), 2),],
)
def test_summary(architecture, backbone, input_shape, n_classes):
    _cfg = cfg.clone()
    _cfg.MODEL.ARCHITECTURE = architecture
    _cfg.MODEL.BACKBONE = backbone
    _cfg.MODEL.FEATURE_SELECTION = False
    MakeModel = get_architecture(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    model = MakeModel(_cfg, logger, input_shape, n_classes)
    model.summary()


@pytest.mark.parametrize(
    "architecture, backbone, input_shape, n_classes, path",
    [("CNN", "VGG16", (200, 200, 3), 2, pytest_output_directory),],
)
def test_save_model(architecture, backbone, input_shape, n_classes, path):
    _cfg = cfg.clone()
    _cfg.MODEL.ARCHITECTURE = architecture
    _cfg.MODEL.BACKBONE = backbone
    _cfg.MODEL.FEATURE_SELECTION = False
    _cfg.OUTPUT_DIR = path
    MakeModel = get_architecture(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    model = MakeModel(_cfg, logger, input_shape, n_classes)
    model.save_model()
    assert os.path.exists(Path(path, "model", "model.json").as_posix())
    assert os.path.exists(Path(path, "model", "model.h5").as_posix())


@pytest.mark.parametrize(
    "architecture, backbone, input_shape, n_classes",
    [("CNN", "VGG16", (200, 200, 3), 2),],
)
def test_get_model(architecture, backbone, input_shape, n_classes):
    _cfg = cfg.clone()
    _cfg.MODEL.ARCHITECTURE = architecture
    _cfg.MODEL.BACKBONE = backbone
    _cfg.MODEL.FEATURE_SELECTION = False
    _cfg.MODEL.FEATURE_SELECTION = False
    MakeModel = get_architecture(_cfg.TASK, _cfg.MODEL.ARCHITECTURE)
    model = MakeModel(_cfg, logger, input_shape, n_classes)
    model.get_model()
