from nnga.utils.data_io import save_model, save_cfg

class BaseNeuralNetwork:
    """ This class implements the Model defined by genetic algorithm indiv
        Parameters
        ----------
            cfg {yacs.config.CfgNode} -- Loaded experiment config

            logger {logging} -- Simple python logging

            input_shape: int
                Inputs shape

            output_dim: int
                Number of outputs

            include_top: bool
                include the top layers

            indiv: Indiv
                Indiv that determine the Model architecture

            keys: List
                List of Indiv keys
        Returns
        -------
    """

    def __init__(self, cfg, logger, input_shape, output_dim, include_top=True,
                 indiv=None, keys=None):
        self._cfg = cfg
        self._logger = logger
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.indiv = indiv
        self.keys = keys
        self.include_top = include_top
        self._model = None

        if self._cfg.MODEL.FEATURE_SELECTION:
            input_dim = 0
            for i in range(sum("feature_selection_" in s for s in self.keys)):
                if self.indiv[self.keys.index(f"feature_selection_{i}")]:
                    input_dim += 1

            if isinstance(self.input_shape, int):
                self.input_shape = input_dim
            else:
                self.input_shape = (input_dim, self.input_shape[1])

        if self._cfg.MODEL.BACKBONE == 'GASearch':
            self.create_model_ga()
            self._logger.info(f"Model {self._cfg.MODEL.ARCHITECTURE} from GA created!")
        else:
            self.create_model()
            if self._cfg.MODEL.FEATURE_SELECTION:
                self._logger.info(f"Model {self._cfg.MODEL.ARCHITECTURE} created with feature selection!")
            else:
                self._logger.info(f"Model {self._cfg.MODEL.ARCHITECTURE} created!")

        if self.include_top:
            self.summary()

    def create_model_ga(self):
        """ This method create the Model defined by genetic algorithm indiv
            Parameters
            ----------

            Returns
            -------

            """
        pass

    def create_model(self):
        """ This method create the Pre Trained Model
            Parameters
            ----------

            Returns
            -------

            """
        pass

    def summary(self):
        """Print the model summary"""
        self._model.summary(print_fn=self._logger.info)

    def save_model(self, seed=''):
        """Save model
        Parameters
            ----------
                seed: int
                    Seed used from GA

            Returns
            -------
        """
        save_model(self._cfg.OUTPUT_DIR, seed, self._model)
        save_cfg(self._cfg.OUTPUT_DIR, seed, self._cfg)

    def get_model(self):
        """Get motel
        Parameters
        ----------

        Returns
        -------
        The model
        """
        return self._model
