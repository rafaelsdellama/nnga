from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    concatenate,
)

from nnga.architectures.base_neural_network import BaseNeuralNetwork
from nnga.architectures.mlp import MLP
from nnga.architectures.cnn import CNN
from nnga.architectures import (
    INICIALIZERS,
    REGULARIZERS,
)
from tensorflow.keras import backend as K


class CNN_MLP(BaseNeuralNetwork):
    """ This class implements the Hibrid Model defined by genetic algorithm indiv
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
                Indiv that determine the Hibrid Model architecture

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
        """ This method create the Hibrid Model defined by genetic algorithm indiv
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

        mlp = MLP(
            self._cfg,
            self._logger,
            self.input_shape[0],
            self.output_dim,
            False,
            self.indiv,
            self.keys,
        ).get_model()
        cnn = CNN(
            self._cfg,
            self._logger,
            self.input_shape[1],
            self.output_dim,
            False,
            self.indiv,
            self.keys,
        ).get_model()

        cnn_mlp = concatenate([mlp.output, cnn.output])

        if self.include_top:
            # Fully connected
            for i in range(sum("units_" in s for s in self.keys)):
                if (
                    i == 0
                    or self.indiv[self.keys.index(f"activate_dense_{i}")]
                ):
                    cnn_mlp = Dense(
                        units=self.indiv[self.keys.index(f"units_{i}")],
                        activation=self.indiv[
                            self.keys.index("activation_dense")
                        ],
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        activity_regularizer=activity_regularizer,
                        bias_regularizer=bias_regularizer,
                    )(cnn_mlp)

                    cnn_mlp = Dropout(
                        self.indiv[self.keys.index(f"dropout_dense_{i}")]
                    )(cnn_mlp)

            cnn_mlp = Dense(
                units=self.output_dim,
                activation="softmax",
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer,
                bias_regularizer=bias_regularizer,
            )(cnn_mlp)

        self._model = Model(inputs=[mlp.input, cnn.input], outputs=cnn_mlp)

    def create_model(self):
        """ This method create the Pre Trained Model
            Parameters
            ----------

            Returns
            -------

            """
        mlp = MLP(
            self._cfg,
            self._logger,
            self.input_shape[0],
            self.output_dim,
            include_top=False,
            indiv=self.indiv,
            keys=self.keys,
        ).get_model()
        cnn = CNN(
            self._cfg,
            self._logger,
            self.input_shape[1],
            self.output_dim,
            include_top=False,
            indiv=self.indiv,
            keys=self.keys,
        ).get_model()

        cnn_mlp = concatenate([mlp.output, cnn.output])

        if self.include_top:
            C = K.int_shape(cnn_mlp)[-1]
            cnn_mlp = Dense(units=max(C, self.output_dim), activation="relu",)(
                cnn_mlp
            )
            cnn_mlp = Dropout(self._cfg.MODEL.DROPOUT)(cnn_mlp)
            cnn_mlp = Dense(
                units=max(C / 2, self.output_dim), activation="relu",
            )(cnn_mlp)
            cnn_mlp = Dropout(self._cfg.MODEL.DROPOUT)(cnn_mlp)
            cnn_mlp = Dense(
                units=max(C / 4, self.output_dim), activation="relu",
            )(cnn_mlp)
            cnn_mlp = Dropout(self._cfg.MODEL.DROPOUT)(cnn_mlp)
            cnn_mlp = Dense(units=self.output_dim, activation="softmax",)(
                cnn_mlp
            )

        self._model = Model(inputs=[mlp.input, cnn.input], outputs=cnn_mlp)
