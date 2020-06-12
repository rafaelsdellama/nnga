from nnga.inference.classification.csv_predictor import CSVPredictor
from nnga.inference.classification.image_predictor import ImagePredictor
from nnga.inference.classification.image_csv_predictor import ImageCSVPredictor
from nnga.inference.segmentation.segmentation_predictor import SegmentationPredictor


PREDICTORS = {
    "Classification": {
        "MLP": CSVPredictor,
        "CNN": ImagePredictor,
        "CNN/MLP": ImageCSVPredictor,
    },
    "Segmentation": {
        "CNN": SegmentationPredictor,
    },
}


def get_predictor(task, backbone):
    """Return a backbone for a specific task and architecture"""
    predictor_by_task = PREDICTORS.get(task)
    predictor = predictor_by_task.get(backbone)
    if predictor_by_task is None:
        raise RuntimeError(
            f"There isn't a valid TASKS!\n Check your experiment config\n"
            f"Tasks: {PREDICTORS.keys()}"
        )

    elif predictor is None:
        raise RuntimeError(
            f"There isn't a valid backbone configured!\nCheck your "
            f"experiment config\nBackbone for {task}: "
            f"{PREDICTORS.get(task).keys()}"
        )
    return predictor