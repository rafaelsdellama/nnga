from nnga.utils.data_io import save_model, save_cfg
import os


class BaseNeuralNetwork:
    """ This class implements the Model defined by genetic algorithm indiv
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
                Indiv that determine the Model architecture

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
        self._cfg = cfg
        self._logger = logger
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.indiv = indiv
        self.keys = keys
        self.include_top = include_top
        self._model = None

        if self._cfg.MODEL.FEATURE_SELECTION:
            input_dim = 0
            for i in range(sum("feature_selection_" in s for s in self.keys)):
                if self.indiv[self.keys.index(f"feature_selection_{i}")]:
                    input_dim += 1

            if input_dim == 0:
                raise ValueError(
                    "The number of features selected should be > 0"
                )

            if isinstance(self.input_shape, int):  # MLP
                self.input_shape = input_dim
            elif (
                isinstance(self.input_shape, tuple)
                and len(self.input_shape) == 2
            ):  # CNN/MLP
                self.input_shape = (input_dim, self.input_shape[1])

        if self._cfg.MODEL.BACKBONE == "GASearch":
            self.create_model_ga()
            self._logger.info(
                f"Model {self._cfg.MODEL.ARCHITECTURE} from GA created!"
            )
        else:
            self.create_model()
            if self._cfg.MODEL.FEATURE_SELECTION:
                self._logger.info(
                    f"Model {self._cfg.MODEL.ARCHITECTURE} "
                    f"created with feature selection!"
                )
            else:
                self._logger.info(
                    f"Model {self._cfg.MODEL.ARCHITECTURE} created!"
                )

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

    def save_model(self):
        """Save model
        """
        save_model(os.path.join(self._cfg.OUTPUT_DIR, "model"), self._model)
        save_cfg(self._cfg.OUTPUT_DIR)

    def get_model(self):
        """Get motel
        Parameters
        ----------

        Returns
        -------
        The model
        """
        return self._model
