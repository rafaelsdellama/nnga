import os
from yacs.config import CfgNode as CN
from nnga.inference import get_predictor


class Predictor:
    """Generalized predictor to NNGA models
        Create the predictor based on cfg.MODEL.ARCHITECTURE

    Parameters
    ----------
        model_dir : {str}
            Path to NNGA folder from model dir.
    """

    def __init__(self, model_dir):
        with open(os.path.join(model_dir, "config.yaml"), "r") as f:
            cfg = CN.load_cfg(f)
            cfg.freeze()

        self._predictor = get_predictor(cfg.TASK, cfg.MODEL.ARCHITECTURE)(
            model_dir, cfg
        )

    def predict(self, sample):
        """Make prediction and return class indexes.

            Parameters
            ----------
                sample : {list}
                    List of {numpy.ndarray}, inputs for model
            Returns
            -------
                tuple :
                    List of {numpy.ndarray} with class indexes, dict to
                    translate class indexes to label name.
        """
        return self._predictor.predict(sample)

    def predict_proba(self, sample):
        """Make prediction and return class probabilities(softmax output).

            Parameters
            ----------
                sample : {list}
                    List of {numpy.ndarray}, inputs for model

            Returns
            -------
                type
                tuple -- List of {numpy.ndarray} with class probabilities,
                dict to translate class indexes to label name.
            """
        return self._predictor.predict_proba(sample)
