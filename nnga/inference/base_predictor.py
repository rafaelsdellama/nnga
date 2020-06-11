import os
import numpy as np
from nnga.utils.data_io import load_decoder_parameters, load_model


class BasePredictor:
    """Base predictor to NNGA models

    Parameters
    ----------
        model_dir : {str}
            Path to NNGA folder from model dir.

        cfg {yacs.config import CfgNode}
            Experiment config data
    """

    def __init__(self, model_dir, cfg):
        self._cfg = cfg
        self._model = load_model(os.path.join(model_dir, "model"))
        self._decode = {
            int(k): v for k, v in load_decoder_parameters(model_dir).items()
        }

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
        return inpt

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
        batch = [self._preprocessing(inpt) for inpt in inpts]
        return np.array(batch)

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
        pred = np.argmax(pred, axis=-1)

        return self._posprocessing(pred), self._decode

    def predict_proba(self, inpts):
        """Make prediction and return class probabilities(softmax output).

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

        return self._posprocessing(pred), self._decode

    def _posprocessing(self, out):
        """Apply posprocessing on predcit output.

            Parameters
            ----------
                out : {numpy.ndarray}
                    Input to be pos-processed

            Returns
            -------
                umpy.ndarray
                    out pos-processed
        """
        return out
