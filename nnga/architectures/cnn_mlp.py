from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, \
    Dropout, BatchNormalization, Input, concatenate

from nnga.architectures.base_neural_network import BaseNeuralNetwork
from nnga.architectures import INICIALIZER, REGULARIZER, create_optimizer


class CNN_MLP(BaseNeuralNetwork):
    """ This class implements the Hibrid Model defined by genetic algorithm indiv
        Parameters
        ----------
            cfg {yacs.config.CfgNode} -- Loaded experiment config

            logger {logging} -- Simple python logging

            datasets: ImageDataset
                dataset to be used in the train/test

            indiv: Indiv
                Indiv that determine the Hibrid Model architecture

            keys: List
                List of Indiv keys
        Returns
        -------
    """

    def __init__(self, cfg, logger, datasets, indiv, keys):
        super().__init__(cfg, logger, datasets, indiv, keys)

    def create_model_ga(self, summary=True):
        """ This method create the Hibrid Model defined by genetic algorithm indiv
            Parameters
            ----------
                summary: bool
                    True: print the Hibrid Model architecture
                    False: does not print the Hibrid Model architecture
            Returns
            -------
                True: if the Hibrid Model architecture is valid
                False: if the Hibrid Model architecture is not valid
            """

        if self._cfg.GA.FEATURE_SELECTION:
            input_dim = 0
            for i in range(sum(
                    'feature_selection_' in s for s in self.keys)):
                if self.indiv[self.keys.index(
                        f'feature_selection_{i}')]:
                    input_dim += 1
        else:
            input_dim = self._datasets['TRAIN']['CSV'].n_features

        input_shape = self._cfg.MODEL.INPUT_SHAPE

        optimizer = create_optimizer(
            self.indiv[self.keys.index('optimizer')],
            self.indiv[self.keys.index('learning_rate')])

        kernel_regularizer = REGULARIZER.get(
            self.indiv[self.keys.index('kernel_regularizer')])
        activity_regularizer = REGULARIZER.get(
            self.indiv[self.keys.index('activity_regularizer')])
        kernel_initializer = INICIALIZER.get(
            self.indiv[self.keys.index('kernel_initializer')])
        bias_regularizer = INICIALIZER.get(
            self.indiv[self.keys.index('bias_regularizer')])

        # create_mlp
        a = Input(shape=(input_dim,))
        mlp = Model(inputs=a, outputs=a)

        try:
            cnn = Sequential()

            cnn.add(Conv2D(
                filters=self.indiv[self.keys.index('filters_0')],
                kernel_size=self.indiv[self.keys.index('kernel_size_0')],
                padding=self.indiv[self.keys.index('padding_0')],
                activation=self.indiv[self.keys.index('activation_cnn')],
                input_shape=input_shape,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer,
                bias_regularizer=bias_regularizer
            ))

            if self.indiv[self.keys.index('batch_normalization_0')]:
                cnn.add(BatchNormalization())

            if self.indiv[self.keys.index(
                    'max_pool_0')]:
                cnn.add(MaxPooling2D(
                    pool_size=self.indiv[self.keys.index('pool_size_0')],
                    padding=self.indiv[self.keys.index('padding_0')]))

            cnn.add(Dropout(self.indiv[self.keys.index(
                'dropout_cnn_0')]))

            # Hidden CNN layers
            for i in range(1, sum(
                    'filters_' in s for s in self.keys)):
                if self.indiv[self.keys.index(
                        f'activate_cnn_{i}')]:
                    cnn.add(Conv2D(
                        filters=self.indiv[self.keys.index(
                            f'filters_{i}')],
                        kernel_size=self.indiv[self.keys.index(
                            f'kernel_size_{i}')],
                        padding=self.indiv[self.keys.index(
                            f'padding_{i}')],
                        activation=self.indiv[self.keys.index(
                            'activation_cnn')],
                        input_shape=input_shape,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        activity_regularizer=activity_regularizer,
                        bias_regularizer=bias_regularizer
                    ))

                    if self.indiv[self.keys.index(
                            f'batch_normalization_{i}')]:
                        cnn.add(BatchNormalization())

                    if self.indiv[self.keys.index(
                            f'max_pool_{i}')]:
                        cnn.add(MaxPooling2D(
                            pool_size=self.indiv[self.keys.index(
                                f'pool_size_{i}')],
                            padding=self.indiv[self.keys.index(
                                f'padding_{i}')]))

                    cnn.add(Dropout(
                        self.indiv[self.keys.index(
                            f'dropout_cnn_{i}')]))

            cnn.add(Flatten())

            cnn_mlp = concatenate([mlp.output, cnn.output])

            # Fully connected
            cnn_mlp = Dense(
                units=self.indiv[self.keys.index(
                    f'units_0')],
                activation=self.indiv[self.keys.index(
                    'activation_dense')],
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer
            )(cnn_mlp)

            cnn_mlp = Dropout(self.indiv[self.keys.index(
                f'dropout_dense_0')])(cnn_mlp)

            for i in range(1, sum('units_' in s for s in self.keys)):
                if self.indiv[self.keys.index(
                        f'activate_dense_{i}')]:
                    cnn_mlp = Dense(
                        units=self.indiv[self.keys.index(
                            f'units_{i}')],
                        activation=self.indiv[self.keys.index(
                            'activation_dense')],
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        activity_regularizer=activity_regularizer
                    )(cnn_mlp)

                    cnn_mlp = Dropout(self.indiv[self.keys.index(
                        f'dropout_dense_{i}')])(cnn_mlp)

            cnn_mlp = Dense(
                units=self._datasets['TRAIN']['IMG'].n_classes,
                activation='softmax',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer
            )(cnn_mlp)

            self._model = Model(
                inputs=[mlp.input, cnn.input],
                outputs=cnn_mlp)

            self._model.compile(
                optimizer=optimizer,
                loss=self._cfg.SOLVER.LOSS,
                metrics=self._cfg.SOLVER.METRICS)

            if summary:
                self._model.summary(print_fn=self._logger.info)
            return True
        except ValueError as e:
            print(e)
            return False
