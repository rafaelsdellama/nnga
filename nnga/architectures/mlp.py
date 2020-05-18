from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input

from nnga.architectures.base_neural_network import BaseNeuralNetwork
from nnga.architectures import (
    INICIALIZERS,
    REGULARIZERS,
)


class MLP(BaseNeuralNetwork):
    """ This class implements the MLP defined by genetic algorithm indiv
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
                Indiv that determine the MLP architecture

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
        output_dim,
        include_top=True,
        indiv=None,
        keys=None,
    ):
        super().__init__(
            cfg, logger, input_shape, output_dim, include_top, indiv, keys
        )

    def create_model_ga(self):
        """ This method create the MLP defined by genetic algorithm indiv
            Parameters
            ----------

            Returns
            -------

        """

        kernel_regularizer = REGULARIZERS.get(
            self.indiv[self.keys.index("kernel_regularizer")]
        )
        activity_regularizer = REGULARIZERS.get(
            self.indiv[self.keys.index("activity_regularizer")]
        )
        kernel_initializer = INICIALIZERS.get(
            self.indiv[self.keys.index("kernel_initializer")]
        )
        bias_regularizer = INICIALIZERS.get(
            self.indiv[self.keys.index("bias_regularizer")]
        )

        input_layer = Input(shape=(self.input_shape,))

        if self.include_top:

            for i in range(sum("units_" in s for s in self.keys)):
                if (
                    i == 0
                    or self.indiv[self.keys.index(f"activate_dense_{i}")]
                ):
                    mlp = Dense(
                        units=self.indiv[self.keys.index(f"units_{i}")],
                        activation=self.indiv[
                            self.keys.index("activation_dense")
                        ],
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        activity_regularizer=activity_regularizer,
                        bias_regularizer=bias_regularizer,
                    )(input_layer if i == 0 else mlp)

                    mlp = Dropout(
                        self.indiv[self.keys.index(f"dropout_dense_{i}")]
                    )(mlp)

            mlp = Dense(
                units=self.output_dim,
                activation="softmax",
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer,
                bias_regularizer=bias_regularizer,
            )(mlp)
            self._model = Model(inputs=input_layer, outputs=mlp)
        else:
            self._model = Model(inputs=input_layer, outputs=input_layer)

    def create_model(self,):
        """ This method create the Pre Trained Model
            Parameters
            ----------

            Returns
            -------

        """

        input_layer = Input(shape=(self.input_shape,))

        if self.include_top:
            mlp = Dense(units=self.input_shape, activation="relu",)(
                input_layer
            )
            mlp = Dense(
                units=max(self.input_shape / 2, self.output_dim),
                activation="relu",
            )(mlp)
            mlp = Dropout(self._cfg.MODEL.DROPOUT)(mlp)
            mlp = Dense(
                units=max(self.input_shape / 4, self.output_dim),
                activation="relu",
            )(mlp)
            mlp = Dropout(self._cfg.MODEL.DROPOUT)(mlp)
            mlp = Dense(units=self.output_dim, activation="softmax",)(mlp)

            self._model = Model(inputs=input_layer, outputs=mlp)
        else:
            self._model = Model(inputs=input_layer, outputs=input_layer)
