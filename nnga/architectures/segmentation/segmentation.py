from nnga.architectures.base_neural_network import BaseNeuralNetwork
from nnga.architectures import get_backbone


class Segmentation(BaseNeuralNetwork):
    """ This class implements the CNN defined by genetic algorithm indiv
        Parameters
        ----------
            cfg {yacs.config.CfgNode} -- Loaded experiment config

            logger {logging} -- Simple python logging

            input_shape: int
                Inputs shape

            n_classes: int
                Number of classes

            include_top: bool
                include the top layers

            indiv: Indiv
                Indiv that determine the CNN architecture

            keys: List
                List of Indiv keys

        Returns
        -------
    """

    def __init__(
        self,
        cfg,
        logger,
        input_shape,
        n_classes,
        include_top=True,
        indiv=None,
        keys=None,
    ):

        super().__init__(
            cfg, logger, input_shape, n_classes, include_top, indiv, keys
        )

    def create_model(self):
        """ This method create the Pre Trained Model
            Parameters
            ----------

            Returns
            -------

        """
        self._model = get_backbone(self._cfg.TASK, self._cfg.MODEL.BACKBONE)(self.input_shape)
