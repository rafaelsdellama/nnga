from nnga.inference.classification.csv_predictor import CSVPredictor
from nnga.inference.classification.image_predictor import ImagePredictor
from nnga.utils.data_manipulation import scale_features
from nnga.utils.data_manipulation import adjust_image_shape, normalize_image
import numpy as np


class ImageCSVPredictor(CSVPredictor, ImagePredictor):
    """Image CSV predictor to NNGA models

    Parameters
    ----------
        model_dir : {str}
            Path to NNGA folder from model dir.

        cfg {yacs.config import CfgNode}
            Experiment config data
    """

    def __init__(self, model_dir, cfg):
        super().__init__(model_dir, cfg)

    def _preprocessing(self, inpt):
        """Apply preprocessing input to be batch and model compatible.

            Parameters
            ----------
                inpt : {numpy.ndarray}
                    Input to be pre-processed

            Returns
            -------
                numpy.ndarray
                    Input pre-processed
        """
        inpt_mlp = scale_features(
            inpt[0],
            self._scale_parameters.keys(),
            self._scale_parameters,
            scale_method=self._cfg.DATASET.SCALER,
        )
        if self._cfg.MODEL.FEATURE_SELECTION:
            inpt_mlp = [
                value
                for i, value in enumerate(inpt_mlp)
                if i in self._feature_selected_idx
            ]

        inpt_cnn = normalize_image(
            adjust_image_shape(
                inpt[1], self._image_shape, self._preserve_ratio,
            )
        )

        return inpt_mlp, inpt_cnn

    def _make_batch(self, inpts):
        """Get a list of inputs and turn it in a batch input.

            This method will be saved and return all information about
            preprocessing to be able that postprocessing return awsers
            with same shape of input for tasks that need it.

            Parameters
            ----------
                inpts : {list}
                    List of inputs, {numpy.ndarray}s with various
                        diferents shapes
            Returns
            -------
                batch input
        """
        return list(
            map(np.array, zip(*[self._preprocessing(inpt) for inpt in inpts]))
        )
