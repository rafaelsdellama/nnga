from nnga.inference.classification.csv_predictor import CSVPredictor
from nnga.inference.classification.image_predictor import ImagePredictor
from nnga.inference.classification.image_csv_predictor import ImageCSVPredictor


PREDICTORS = {
    "MLP": CSVPredictor,
    "CNN": ImagePredictor,
    "CNN/MLP": ImageCSVPredictor,
}
