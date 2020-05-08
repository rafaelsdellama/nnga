from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from nnga.architectures.base_neural_network import BaseNeuralNetwork
from nnga.architectures import INICIALIZER, REGULARIZER, create_optimizer


class MLP(BaseNeuralNetwork):
    """ This class implements the MLP defined by genetic algorithm indiv
        Parameters
        ----------
            cfg {yacs.config.CfgNode} -- Loaded experiment config

            logger {logging} -- Simple python logging

            datasets: ImageDataset
                dataset to be used in the train/test

            indiv: Indiv
                Indiv that determine the MLP architecture

            keys: List
                List of Indiv keys
        Returns
        -------
    """

    def __init__(self, cfg, logger, datasets, indiv, keys):
        super().__init__(cfg, logger, datasets, indiv, keys)

    def create_model_ga(self, summary=True):
        """ This method create the MLP defined by genetic algorithm indiv
            Parameters
            ----------
                summary: bool
                    True: print the MLP architecture
                    False: does not print the MLP architecture
            Returns
            -------
                True: if the MLP architecture is valid
                False: if the MLP architecture is not valid
            """

        if self._cfg.GA.FEATURE_SELECTION:
            input_dim = 0
            for i in range(sum("feature_selection_" in s for s in self.keys)):
                if self.indiv[self.keys.index(f"feature_selection_{i}")]:
                    input_dim += 1
        else:
            input_dim = self._datasets["TRAIN"]["CSV"].n_features

        optimizer = create_optimizer(
            self.indiv[self.keys.index("optimizer")],
            self.indiv[self.keys.index("learning_rate")],
        )

        kernel_regularizer = REGULARIZER.get(
            self.indiv[self.keys.index("kernel_regularizer")]
        )
        activity_regularizer = REGULARIZER.get(
            self.indiv[self.keys.index("activity_regularizer")]
        )
        kernel_initializer = INICIALIZER.get(
            self.indiv[self.keys.index("kernel_initializer")]
        )
        bias_regularizer = INICIALIZER.get(
            self.indiv[self.keys.index("bias_regularizer")]
        )

        try:
            self._model = Sequential()

            self._model.add(
                Dense(
                    units=self.indiv[self.keys.index("units_0")],
                    activation=self.indiv[self.keys.index("activation_dense")],
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=activity_regularizer,
                    bias_regularizer=bias_regularizer,
                    input_dim=input_dim,
                )
            )

            self._model.add(
                Dropout(self.indiv[self.keys.index("dropout_dense_0")])
            )

            for i in range(1, sum("units_" in s for s in self.keys)):
                if self.indiv[self.keys.index(f"activate_dense_{i}")]:
                    self._model.add(
                        Dense(
                            units=self.indiv[self.keys.index(f"units_{i}")],
                            activation=self.indiv[
                                self.keys.index("activation_dense")
                            ],
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            activity_regularizer=activity_regularizer,
                            bias_regularizer=bias_regularizer,
                        )
                    )

                    self._model.add(
                        Dropout(
                            self.indiv[self.keys.index(f"dropout_dense_{i}")]
                        )
                    )

            self._model.add(
                Dense(
                    units=self._datasets["TRAIN"]["CSV"].n_classes,
                    activation="softmax",
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=activity_regularizer,
                    bias_regularizer=bias_regularizer,
                )
            )

            self._model.compile(
                optimizer=optimizer,
                loss=self._cfg.SOLVER.LOSS,
                metrics=self._cfg.SOLVER.METRICS,
            )

            if summary:
                self._model.summary(print_fn=self._logger.info)
            return True
        except ValueError as e:
            print(e)
            return False
