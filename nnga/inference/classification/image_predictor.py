from nnga.inference.base_predictor import BasePredictor
from nnga.utils.data_manipulation import adjust_image_shape, normalize_image


class ImagePredictor(BasePredictor):
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
            adjust_image_shape(inpt, self._image_shape, self._preserve_ratio,)
        )
