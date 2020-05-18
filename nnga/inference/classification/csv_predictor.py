from nnga.inference.base_predictor import BasePredictor
from nnga.utils.data_manipulation import scale_features
from nnga.utils.data_io import load_scale_parameters, load_feature_selected


class CSVPredictor(BasePredictor):
    """CSV predictor to NNGA models

    Parameters
    ----------
        model_dir : {str}
            Path to NNGA folder from model dir.

        cfg {yacs.config import CfgNode}
            Experiment config data
    """

    def __init__(self, model_dir, cfg):
        super().__init__(model_dir, cfg)

        self._scale_parameters = load_scale_parameters(model_dir)

        if cfg.MODEL.FEATURE_SELECTION:
            self._feature_selected = load_feature_selected(model_dir)
            self._feature_selected_idx = [
                i
                for i, value in enumerate(self._scale_parameters.keys())
                if value in self._feature_selected
            ]

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
        inpt = scale_features(
            inpt,
            self._scale_parameters.keys(),
            self._scale_parameters,
            scale_method=self._cfg.DATASET.SCALER,
        )
        if self._cfg.MODEL.FEATURE_SELECTION:
            inpt = [
                value
                for i, value in enumerate(inpt)
                if i in self._feature_selected_idx
            ]

        return inpt
