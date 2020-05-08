from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
)

from nnga.architectures.base_neural_network import BaseNeuralNetwork
from nnga.architectures import INICIALIZER, REGULARIZER, create_optimizer


class CNN(BaseNeuralNetwork):
    """ This class implements the CNN defined by genetic algorithm indiv
        Parameters
        ----------
            cfg {yacs.config.CfgNode} -- Loaded experiment config

            logger {logging} -- Simple python logging

            datasets: ImageDataset
                dataset to be used in the train/test

            indiv: Indiv
                Indiv that determine the CNN architecture

            keys: List
                List of Indiv keys
        Returns
        -------
    """

    def __init__(self, cfg, logger, datasets, indiv, keys):
        super().__init__(cfg, logger, datasets, indiv, keys)

    def create_model_ga(self, summary=True):
        """ This method create the CNN defined by genetic algorithm indiv
            Parameters
            ----------
                summary: bool
                    True: print the CNN architecture
                    False: does not print the CNN architecture
            Returns
            -------
                True: if the CNN architecture is valid
                False: if the CNN architecture is not valid
            """

        input_shape = self._cfg.MODEL.INPUT_SHAPE

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
                Conv2D(
                    filters=self.indiv[self.keys.index("filters_0")],
                    kernel_size=self.indiv[self.keys.index("kernel_size_0")],
                    padding=self.indiv[self.keys.index("padding_0")],
                    activation=self.indiv[self.keys.index("activation_cnn")],
                    input_shape=input_shape,
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=activity_regularizer,
                    bias_regularizer=bias_regularizer,
                )
            )

            if self.indiv[self.keys.index("batch_normalization_0")]:
                self._model.add(BatchNormalization())

            if self.indiv[self.keys.index("max_pool_0")]:
                self._model.add(
                    MaxPooling2D(
                        pool_size=self.indiv[self.keys.index("pool_size_0")],
                        padding=self.indiv[self.keys.index("padding_0")],
                    )
                )

            self._model.add(
                Dropout(self.indiv[self.keys.index("dropout_cnn_0")])
            )

            # Hidden CNN layers
            for i in range(1, sum("filters_" in s for s in self.keys)):
                if self.indiv[self.keys.index(f"activate_cnn_{i}")]:
                    self._model.add(
                        Conv2D(
                            filters=self.indiv[
                                self.keys.index(f"filters_{i}")
                            ],
                            kernel_size=self.indiv[
                                self.keys.index(f"kernel_size_{i}")
                            ],
                            padding=self.indiv[
                                self.keys.index(f"padding_{i}")
                            ],
                            activation=self.indiv[
                                self.keys.index("activation_cnn")
                            ],
                            input_shape=input_shape,
                            kernel_initializer=kernel_initializer,
                            kernel_regularizer=kernel_regularizer,
                            activity_regularizer=activity_regularizer,
                            bias_regularizer=bias_regularizer,
                        )
                    )

                    if self.indiv[self.keys.index(f"batch_normalization_{i}")]:
                        self._model.add(BatchNormalization())

                    if self.indiv[self.keys.index(f"max_pool_{i}")]:
                        self._model.add(
                            MaxPooling2D(
                                pool_size=self.indiv[
                                    self.keys.index(f"pool_size_{i}")
                                ],
                                padding=self.indiv[
                                    self.keys.index(f"padding_{i}")
                                ],
                            )
                        )

                    self._model.add(
                        Dropout(
                            self.indiv[self.keys.index(f"dropout_cnn_{i}")]
                        )
                    )

            self._model.add(Flatten())

            # Fully connected
            self._model.add(
                Dense(
                    units=self.indiv[self.keys.index(f"units_0")],
                    activation=self.indiv[self.keys.index("activation_dense")],
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=activity_regularizer,
                )
            )

            self._model.add(
                Dropout(self.indiv[self.keys.index(f"dropout_dense_0")])
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
                        )
                    )

                    self._model.add(
                        Dropout(
                            self.indiv[self.keys.index(f"dropout_dense_{i}")]
                        )
                    )

            self._model.add(
                Dense(
                    units=self._datasets["TRAIN"]["IMG"].n_classes,
                    activation="softmax",
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=activity_regularizer,
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
