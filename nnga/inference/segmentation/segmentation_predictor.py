import cv2
import numpy as np
from nnga.inference.base_predictor import BasePredictor
from nnga.utils.data_manipulation import adjust_image_shape, normalize_image


class SegmentationPredictor(BasePredictor):
    """Image predictor to NNGA models

    Parameters
    ----------
        model_dir : {str}
            Path to NNGA folder from model dir.

        cfg {yacs.config import CfgNode}
            Experiment config data
    """

    def __init__(self, model_dir, cfg):
        super().__init__(model_dir, cfg)
        self._image_shape = tuple(cfg.MODEL.INPUT_SHAPE)
        self._preserve_ratio = cfg.DATASET.PRESERVE_IMG_RATIO

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
        return normalize_image(
            adjust_image_shape(inpt, self._image_shape,)
        )

    def predict(self, inpts):
        """Make prediction and return class indexes.

            Parameters
            ----------
                inpts : {list}
                    List of {numpy.ndarray}, inputs for model
            Returns
            -------
                type
                tuple -- List of {numpy.ndarray} with class probabilities,
                dict to translate class indexes to label name.
        """
        batch = self._make_batch(inpts)

        pred = self._model.predict(batch)

        masks = []
        for i in range(len(pred)):
            predcit_mask = pred[i]
            _, predcit_mask = cv2.threshold(predcit_mask, 0.5, 255, cv2.THRESH_BINARY)
            masks.append(predcit_mask.astype('uint8'))

        return self._posprocessing(np.array(masks)), self._decode
