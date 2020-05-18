from nnga.inference.predictor import Predictor
from nnga.utils.data_io import load_image
from nnga import ROOT
from pathlib import Path
import pandas as pd
import numpy as np
import pytest


test_directory = Path(ROOT, "tests", "testdata").as_posix()
features_mnist = Path(
    test_directory, "datasets", "mnist", "features.csv"
).as_posix()
img_mnist = Path(test_directory, "datasets", "mnist", "0", "3.png").as_posix()
model_dir = Path(test_directory, "models").as_posix()

df_test = pd.read_csv(features_mnist)
img = load_image(img_mnist)


@pytest.mark.parametrize(
    "model", ["mlp", "mlp_feature", "mlp_feature_GASearch", "mlp_GASearch"],
)
def test_mlp_predictor_predict(model):
    mlp_model_dir = Path(model_dir, model).as_posix()
    nnga_predictor = Predictor(mlp_model_dir)
    pred, decode = nnga_predictor.predict(
        [df_test.iloc[0].drop(["class", "id"]).values]
    )
    assert isinstance(pred, np.ndarray)
    assert len(pred) == 1
    assert isinstance(decode, dict)
    assert len(decode) == 10


@pytest.mark.parametrize(
    "model", ["mlp", "mlp_feature", "mlp_feature_GASearch", "mlp_GASearch"],
)
def test_mlp_predictor_predict_proba(model):
    mlp_model_dir = Path(model_dir, model).as_posix()
    nnga_predictor = Predictor(mlp_model_dir)
    pred, decode = nnga_predictor.predict_proba(
        [df_test.iloc[0].drop(["class", "id"]).values]
    )
    assert isinstance(pred, np.ndarray)
    assert len(pred) == 1
    assert isinstance(decode, dict)
    assert len(decode) == 10


@pytest.mark.parametrize(
    "model", ["cnn_GASearch"],
)
def test_cnn_predictor_predict(model):
    cnn_model_dir = Path(model_dir, model).as_posix()
    nnga_predictor = Predictor(cnn_model_dir)
    pred, decode = nnga_predictor.predict([img])
    assert isinstance(pred, np.ndarray)
    assert len(pred) == 1
    assert isinstance(decode, dict)
    assert len(decode) == 10


@pytest.mark.parametrize(
    "model", ["cnn_GASearch"],
)
def test_cnn_predictor_predict_proba(model):
    cnn_model_dir = Path(model_dir, model).as_posix()
    nnga_predictor = Predictor(cnn_model_dir)
    pred, decode = nnga_predictor.predict_proba([img])
    assert isinstance(pred, np.ndarray)
    assert len(pred) == 1
    assert isinstance(decode, dict)
    assert len(decode) == 10


@pytest.mark.parametrize(
    "model", ["cnn_mlp_feature_GASearch", "cnn_mlp_GASearch",],
)
def test_cnn_mlp_predictor_predict(model):
    cnn_mlp_model_dir = Path(model_dir, model).as_posix()
    nnga_predictor = Predictor(cnn_mlp_model_dir)
    pred, decode = nnga_predictor.predict(
        [[df_test.iloc[0].drop(["class", "id"]).values, img]]
    )
    assert isinstance(pred, np.ndarray)
    assert len(pred) == 1
    assert isinstance(decode, dict)
    assert len(decode) == 10


@pytest.mark.parametrize(
    "model", ["cnn_mlp_feature_GASearch", "cnn_mlp_GASearch",],
)
def test_cnn_mlp_predictor_predict_proba(model):
    cnn_mlp_model_dir = Path(model_dir, model).as_posix()
    nnga_predictor = Predictor(cnn_mlp_model_dir)
    pred, decode = nnga_predictor.predict_proba(
        [[df_test.iloc[0].drop(["class", "id"]).values, img]]
    )
    assert isinstance(pred, np.ndarray)
    assert len(pred) == 1
    assert isinstance(decode, dict)
    assert len(decode) == 10
