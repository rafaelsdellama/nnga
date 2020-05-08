import pytest
from nnga import ROOT
from pathlib import Path
import random as rd
import numpy as np
from nnga.genetic_algorithm.population import Population
from nnga.utils.metrics import compute_metrics
from nnga.utils.data_io import (
    load_image,
    load_csv_file,
    load_statistic,
    load_pop,
    save_pop,
    save_statistic,
    save_history,
    save_roc_curve,
    save_metrics,
)


test_directory = Path(ROOT, "tests", "testdata").as_posix()

img_test = Path(
    test_directory, "img_dataset", "Class_1", "class1_001.png"
).as_posix()
iris = Path(test_directory, "iris.csv").as_posix()
iris2 = Path(test_directory, "iris2.csv").as_posix()
iris_cols = [
    "sepal length",
    "sepal width",
    "petal length",
    "petal width",
    "class",
]
pop_directory = Path(test_directory, "iris_test").as_posix()
pytest_output_directory = "./Pytest_output"


@pytest.mark.parametrize(
    "path, input_shape, preserve_ratio",
    [
        (img_test, (100, 100), True),
        (img_test, (100, 100), False),
        (img_test, (300, 150), True),
        (img_test, (150, 300), False),
        (img_test, (100, 100, 3), True),
        (img_test, (100, 100, 3), False),
        (img_test, (300, 150, 3), True),
        (img_test, (150, 300, 3), False),
    ],
)
def test_load_image(path, input_shape, preserve_ratio):
    img = load_image(path, input_shape, preserve_ratio)
    assert input_shape == img.shape


@pytest.mark.parametrize(
    "path, input_shape, preserve_ratio, error",
    [
        ("wrong_path", (100, 100), True, FileNotFoundError),
        (3, (100, 100), False, TypeError),
    ],
)
def test_load_image_wrong_path(path, input_shape, preserve_ratio, error):
    with pytest.raises(error):
        load_image(path, input_shape, preserve_ratio)


@pytest.mark.parametrize(
    "path, usecols, chunksize",
    [
        (iris, iris_cols, None),
        (iris, iris_cols[0:3], None),
        (iris, iris_cols, 5),
        (iris2, iris_cols, None),
        (iris2, iris_cols[0:3], None),
        (iris2, iris_cols, 5),
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
    "path, usecols, chunksize, error",
    [
        ("wrong_path", iris_cols, None, FileNotFoundError),
        (3, iris_cols, None, TypeError),
    ],
)
def test_load_csv_file_wrong_path(path, usecols, chunksize, error):
    with pytest.raises(error):
        load_csv_file(path, usecols, chunksize)


@pytest.mark.parametrize(
    "path, seed", [(pop_directory, 0)],
)
def test_load_statistic(path, seed):
    df = load_statistic(path, seed)
    assert len(df.shape) == 2
    assert df.shape[1] == 3


@pytest.mark.parametrize(
    "path, seed, error", [(pop_directory, 1, FileNotFoundError)],
)
def test_load_statistic_wrong_path(path, seed, error):
    with pytest.raises(error):
        load_statistic(path, seed)


@pytest.mark.parametrize(
    "path, seed", [(pop_directory, 0)],
)
def test_load_pop(path, seed):
    df = load_pop(path, seed)
    assert len(df.shape) == 2
    assert df.shape[1] == 2


@pytest.mark.parametrize(
    "path, seed, error", [(pop_directory, 1, FileNotFoundError)],
)
def test_load_pop_wrong_path(path, seed, error):
    with pytest.raises(error):
        load_pop(path, seed)


@pytest.mark.parametrize(
    "path, seed, pop", [(pytest_output_directory, 0, Population(5))],
)
def test_save_pop(path, seed, pop):
    save_pop(path, seed, pop)


@pytest.mark.parametrize(
    "path, seed", [(pytest_output_directory, 0)],
)
def test_save_statistic(path, seed):
    statistic = []

    for _ in range(5):
        statistic.append(
            {
                "mean_fitness": rd.random(),
                "best_fitness": rd.random(),
                "best_indiv": rd.randint(0, 5),
            }
        )

    save_statistic(path, seed, statistic)


@pytest.mark.parametrize(
    "path, seed, acc_key, val_acc_key",
    [
        (pytest_output_directory, 0, "acc", "val_acc"),
        (pytest_output_directory, 0, "accuracy", "val_accuracy"),
        (
            pytest_output_directory,
            0,
            "categorical_accuracy",
            "val_categorical_accuracy",
        ),
    ],
)
def test_save_history(path, seed, acc_key, val_acc_key):
    history = {
        acc_key: list(np.random.rand(5)),
        "loss": list(np.random.uniform(low=0.0, high=5.0, size=5)),
        val_acc_key: list(np.random.rand(5)),
        "val_loss": list(np.random.uniform(low=0.0, high=5.0, size=5)),
    }

    save_history(path, seed, history)


@pytest.mark.parametrize(
    "path, seed, lbl, predict_proba, labels",
    [
        (
            pytest_output_directory,
            0,
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
def test_save_roc_curve(path, seed, lbl, predict_proba, labels):
    save_roc_curve(path, seed, lbl, predict_proba, labels)


@pytest.mark.parametrize(
    "path, seed", [(pytest_output_directory, 0)],
)
def test_save_metrics(path, seed):
    metrics = compute_metrics(
        [0, 0, 1],
        [1, 0, 0],
        np.array(
            [np.array([0.1, 0.9]), np.array([0.8, 0.2]), np.array([0.7, 0.3])]
        ),
        ["Class 0", "Class 1"],
        ["0", "1", "2"],
    )
    save_metrics(path, seed, metrics)
