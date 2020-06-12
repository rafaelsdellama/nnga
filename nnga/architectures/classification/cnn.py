from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
    # GlobalAveragePooling2D,
    # GlobalMaxPooling2D
)

from nnga.architectures.base_neural_network import BaseNeuralNetwork
from nnga.architectures import (
    get_backbone,
    INICIALIZERS,
    REGULARIZERS,
)
from tensorflow.keras import backend as K


class CNN(BaseNeuralNetwork):
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

    def create_model_ga(self):
        """ This method create the CNN defined by genetic algorithm indiv
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

        input_layer = Input(shape=self.input_shape)

        for i in range(sum("filters_" in s for s in self.keys)):
            if i == 0 or self.indiv[self.keys.index(f"activate_cnn_{i}")]:
                cnn = Conv2D(
                    filters=self.indiv[self.keys.index(f"filters_{i}")],
                    kernel_size=self.indiv[
                        self.keys.index(f"kernel_size_{i}")
                    ],
                    padding=self.indiv[self.keys.index(f"padding_{i}")],
                    activation=self.indiv[self.keys.index("activation_cnn")],
                    kernel_initializer=kernel_initializer,
                    kernel_regularizer=kernel_regularizer,
                    activity_regularizer=activity_regularizer,
                    bias_regularizer=bias_regularizer,
                )(input_layer if i == 0 else cnn)

                if self.indiv[self.keys.index(f"batch_normalization_{i}")]:
                    cnn = BatchNormalization()(cnn)

                if self.indiv[self.keys.index(f"max_pool_{i}")]:
                    cnn = MaxPooling2D(
                        pool_size=self.indiv[
                            self.keys.index(f"pool_size_{i}")
                        ],
                        padding=self.indiv[self.keys.index(f"padding_{i}")],
                    )(cnn)

                cnn = Dropout(self.indiv[self.keys.index(f"dropout_cnn_{i}")])(
                    cnn
                )

        cnn = Flatten()(cnn)

        if self.include_top:
            # Fully connected
            for i in range(sum("units_" in s for s in self.keys)):
                if (
                    i == 0
                    or self.indiv[self.keys.index(f"activate_dense_{i}")]
                ):
                    cnn = Dense(
                        units=self.indiv[self.keys.index(f"units_{i}")],
                        activation=self.indiv[
                            self.keys.index("activation_dense")
                        ],
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        activity_regularizer=activity_regularizer,
                    )(cnn)

                    cnn = Dropout(
                        self.indiv[self.keys.index(f"dropout_dense_{i}")]
                    )(cnn)

            cnn = Dense(
                units=self.n_classes,
                activation="softmax",
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer,
            )(cnn)

        self._model = Model(inputs=input_layer, outputs=cnn)

    def create_model(self):
        """ This method create the Pre Trained Model
            Parameters
            ----------

            Returns
            -------

        """
        base_model = get_backbone(self._cfg.TASK, self._cfg.MODEL.BACKBONE)(
            include_top=False, input_shape=self.input_shape
        )

        # Freeze the layers except the last 4 layers
        # for layer in base_model.layers[:-4]:
        #     layer.trainable = False

        # mlp = GlobalAveragePooling2D()(base_model.output)
        # mlp = GlobalMaxPooling2D()(base_model.output)
        mlp = Flatten()(base_model.output)

        if self.include_top:
            C = K.int_shape(mlp)[-1]
            mlp = Dense(units=max(C / 2, self.n_classes), activation="relu",)(mlp)
            mlp = Dropout(self._cfg.MODEL.DROPOUT)(mlp)
            mlp = Dense(units=max(C / 4, self.n_classes), activation="relu",)(
                mlp
            )
            mlp = Dropout(self._cfg.MODEL.DROPOUT)(mlp)
            mlp = Dense(units=max(C / 8, self.n_classes), activation="relu",)(
                mlp
            )
            mlp = Dropout(self._cfg.MODEL.DROPOUT)(mlp)
            mlp = Dense(units=self.n_classes, activation="softmax",)(mlp)

        self._model = Model(inputs=base_model.input, outputs=mlp)
