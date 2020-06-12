import pytest
from nnga import ROOT
from pathlib import Path
import random as rd
import numpy as np
import os
from nnga.genetic_algorithm.population import Population
from nnga.utils.metrics import compute_metrics
from nnga.utils.data_io import (
    load_statistic,
    load_pop,
    save_statistic,
    save_pop,
    save_history,
    save_cfg,
    save_roc_curve,
    save_metrics,
    load_csv_file,
    load_csv_line,
    load_image,
    save_scale_parameters,
    save_feature_selected,
    save_encoder_parameters,
    save_decoder_parameters,
    load_scale_parameters,
    load_feature_selected,
    load_encoder_parameters,
    load_decoder_parameters,
    load_model,
)


test_directory = Path(ROOT, "tests", "testdata").as_posix()

img_test = Path(test_directory, "datasets", "classification", "mnist",
                "0", "3.png").as_posix()
img_features = Path(
    test_directory, "datasets", "classification", "mnist", "features.csv"
).as_posix()
img_features2 = Path(
    test_directory, "datasets", "classification", "mnist", "features.csv"
).as_posix()
img_features_cols = [
    "class",
    "id",
    "feature_0",
    "feature_1",
    "feature_2",
    "feature_3",
    "feature_4",
    "feature_5",
    "feature_6",
    "feature_7",
    "feature_8",
    "feature_9",
    "feature_10",
    "feature_11",
    "feature_12",
    "feature_13",
    "feature_14",
    "feature_15",
]
model_saved_dir = Path(
    test_directory, "models", "cnn_mlp_feature_GASearch"
).as_posix()
model_saved_file = Path(
    test_directory, "models", "cnn_mlp_feature_GASearch", "model"
).as_posix()
pytest_output_directory = "./Pytest_output"


@pytest.mark.parametrize(
    "path", [img_test,],
)
def test_load_image(path):
    img = load_image(path)
    assert len(img.shape) == 3


@pytest.mark.parametrize(
    "path, error", [("wrong_path", FileNotFoundError), (3, TypeError),],
)
def test_load_image_wrong_path(path, error):
    with pytest.raises(error):
        load_image(path)


@pytest.mark.parametrize(
    "path, usecols, chunksize",
    [
        (img_features, img_features_cols, None),
        (img_features, img_features_cols[0:3], None),
        (img_features, img_features_cols, 5),
        (img_features2, img_features_cols, None),
        (img_features2, img_features_cols[0:3], None),
        (img_features2, img_features_cols, 5),
    ],
)
def test_load_csv_file(path, usecols, chunksize):
    df = load_csv_file(path, usecols, chunksize)

    if chunksize is None:
        assert len(df.shape) == 2
        assert df.shape[1] == len(usecols)
    else:
        assert len(df.get_chunk().shape) == 2
        assert df.get_chunk().shape[0] == chunksize
        assert df.get_chunk().shape[1] == len(usecols)


@pytest.mark.parametrize(
    "path, sep, idx", [(img_features, ",", 0), (img_features2, ";", 0),],
)
def test_load_csv_line(path, sep, idx):
    sample = load_csv_line(path, sep, idx)

    assert isinstance(sample, tuple) and len(sample) == 2
    assert isinstance(sample[0], list)
    assert len(sample[0]) > 0
    assert isinstance(sample[1], list)
    assert len(sample[1]) > 0


@pytest.mark.parametrize(
    "path, usecols, chunksize, error",
    [
        ("wrong_path", img_features_cols, None, FileNotFoundError),
        (3, img_features_cols, None, TypeError),
    ],
)
def test_load_csv_file_wrong_path(path, usecols, chunksize, error):
    with pytest.raises(error):
        load_csv_file(path, usecols, chunksize)


@pytest.mark.parametrize(
    "path", [model_saved_dir],
)
def test_load_statistic(path):
    df = load_statistic(path)
    assert len(df.shape) == 2
    assert df.shape[1] == 3


@pytest.mark.parametrize(
    "path, error", [("wrong", FileNotFoundError)],
)
def test_load_statistic_wrong_path(path, error):
    with pytest.raises(error):
        load_statistic(path)


@pytest.mark.parametrize(
    "path", [model_saved_dir],
)
def test_load_pop(path):
    df = load_pop(path)
    assert len(df.shape) == 2
    assert df.shape[1] == 2


@pytest.mark.parametrize(
    "path, error", [("wrong", FileNotFoundError)],
)
def test_load_pop_wrong_path(path, error):
    with pytest.raises(error):
        load_pop(path)


@pytest.mark.parametrize(
    "path, pop", [(pytest_output_directory, Population(5))],
)
def test_save_pop(path, pop):
    save_pop(path, pop)
    assert os.path.exists(Path(path, "ga", "pop.csv").as_posix())


@pytest.mark.parametrize(
    "path", [pytest_output_directory],
)
def test_save_statistic(path):
    statistic = []

    for _ in range(5):
        statistic.append(
            {
                "mean_fitness": rd.random(),
                "best_fitness": rd.random(),
                "best_indiv": rd.randint(0, 5),
            }
        )

    save_statistic(path, statistic)
    assert os.path.exists(Path(path, "ga", "results.csv").as_posix())
    assert os.path.exists(Path(path, "ga", "Generations.png").as_posix())


@pytest.mark.parametrize(
    "path, acc_key, val_acc_key",
    [
        (pytest_output_directory, "acc", "val_acc"),
        (pytest_output_directory, "accuracy", "val_accuracy"),
        (
            pytest_output_directory,
            "categorical_accuracy",
            "val_categorical_accuracy",
        ),
    ],
)
def test_save_history(path, acc_key, val_acc_key):
    history = {
        acc_key: list(np.random.rand(5)),
        "loss": list(np.random.uniform(low=0.0, high=5.0, size=5)),
        val_acc_key: list(np.random.rand(5)),
        "val_loss": list(np.random.uniform(low=0.0, high=5.0, size=5)),
    }

    save_history(path, history)
    assert os.path.exists(
        Path(path, "metrics", "history_model.png").as_posix()
    )


@pytest.mark.parametrize(
    "path, lbl, predict_proba, labels",
    [
        (
            pytest_output_directory,
            [0, 0, 1],
            np.array(
                [
                    np.array([0.1, 0.9]),
                    np.array([0.8, 0.2]),
                    np.array([0.7, 0.3]),
                ]
            ),
            ["Class 0", "Class 1"],
        ),
    ],
)
def test_save_roc_curve(path, lbl, predict_proba, labels):
    save_roc_curve(path, lbl, predict_proba, labels)
    assert os.path.exists(Path(path, "metrics", "roc_curve.png").as_posix())


@pytest.mark.parametrize(
    "path", [pytest_output_directory],
)
def test_save_metrics(path):
    metrics = compute_metrics(
        [0, 0, 1],
        [1, 0, 0],
        np.array(
            [np.array([0.1, 0.9]), np.array([0.8, 0.2]), np.array([0.7, 0.3])]
        ),
        ["Class 0", "Class 1"],
        ["0", "1", "2"],
    )
    save_metrics(path, metrics)
    assert os.path.exists(Path(path, "metrics", "metrics.txt").as_posix())


@pytest.mark.parametrize(
    "path", [pytest_output_directory],
)
def test_save_cfg(path):
    save_cfg(path)
    assert os.path.exists(Path(path, "config.yaml").as_posix())


@pytest.mark.parametrize(
    "path, scale_parameters",
    [(pytest_output_directory, {"FeatureX": {"min": 0, "max": 1}})],
)
def test_save_scale_parameters(path, scale_parameters):
    save_scale_parameters(path, scale_parameters)
    assert os.path.exists(
        Path(path, "dataset", "scale_parameters.json").as_posix()
    )


@pytest.mark.parametrize(
    "path, features", [(pytest_output_directory, ["Feature_1", "Feature_2"])],
)
def test_save_feature_selected(path, features):
    save_feature_selected(path, features)
    assert os.path.exists(
        Path(path, "dataset", "feature_selected.json").as_posix()
    )


@pytest.mark.parametrize(
    "path, encode", [(pytest_output_directory, {"class_0": 0})],
)
def test_save_encoder_parameters(path, encode):
    save_encoder_parameters(path, encode)
    assert os.path.exists(Path(path, "dataset", "encode.json").as_posix())


@pytest.mark.parametrize(
    "path, decode", [(pytest_output_directory, {0: "class_0"})],
)
def test_save_decoder_parameters(path, decode):
    save_decoder_parameters(path, decode)
    assert os.path.exists(Path(path, "dataset", "decode.json").as_posix())


@pytest.mark.parametrize(
    "path", [model_saved_dir],
)
def test_load_scale_parameters(path):
    scale_parameters = load_scale_parameters(path)
    assert isinstance(scale_parameters, dict)
    assert len(scale_parameters) > 0
    assert isinstance(scale_parameters[list(scale_parameters.keys())[0]], dict)


@pytest.mark.parametrize(
    "path", [model_saved_dir],
)
def test_load_feature_selected(path):
    feature_selected = load_feature_selected(path)
    assert isinstance(feature_selected, list)
    assert len(feature_selected) > 0


@pytest.mark.parametrize(
    "path", [model_saved_dir],
)
def test_load_encoder_parameters(path):
    encoder = load_encoder_parameters(path)
    assert isinstance(encoder, dict)
    assert len(encoder) > 0


@pytest.mark.parametrize(
    "path", [model_saved_dir],
)
def test_load_decoder_parameters(path):
    decoder = load_decoder_parameters(path)
    assert isinstance(decoder, dict)
    assert len(decoder) > 0


@pytest.mark.parametrize(
    "path", [model_saved_file],
)
def test_load_model(path):
    model = load_model(path)
    assert model is not None
