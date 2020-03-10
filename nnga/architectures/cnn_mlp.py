from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, \
    Dropout, BatchNormalization, Input, concatenate

from nnga.architectures.base_neural_network import BaseNeuralNetwork
from nnga.architectures import INICIALIZER, REGULARIZER, create_optimizer


class CNN_MLP(BaseNeuralNetwork):

    def __init__(self, cfg, logger, datasets, indiv, keys, seed):
        super().__init__(cfg, logger, datasets, indiv, keys, seed)

    def create_model_ga(self):
        if self._cfg.GA.ATTRIBUTE_SELECTION:
            input_dim = 0
            for i in range(sum(
                    'attribute_selection_' in s for s in self.keys)):
                if self.indiv[self.keys.index(
                        'attribute_selection_' + str(i))]:
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
                activation=self.indiv[self.keys.index('activation_conv')],
                input_shape=input_shape,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer,
                bias_regularizer=bias_regularizer
            ))

            if self.indiv[self.keys.index('batchNormalization_0')]:
                cnn.add(BatchNormalization())

            if self.indiv[self.keys.index(
                    'maxPool_0')]:
                cnn.add(MaxPooling2D(
                    pool_size=self.indiv[self.keys.index('pool_size_0')],
                    padding=self.indiv[self.keys.index('padding_0')]))

            cnn.add(Dropout(self.indiv[self.keys.index(
                'dropout_cnn_0')]))

            # Hidden CNN layers
            for i in range(sum(
                    'activate_cnn_' in s for s in self.keys)):
                if self.indiv[self.keys.index(
                        'activate_cnn_' + str(i + 1))]:
                    cnn.add(Conv2D(
                        filters=self.indiv[self.keys.index(
                            'filters_' + str(i + 1))],
                        kernel_size=self.indiv[self.keys.index(
                            'kernel_size_' + str(i + 1))],
                        padding=self.indiv[self.keys.index(
                            'padding_' + str(i + 1))],
                        activation=self.indiv[self.keys.index(
                            'activation_conv')],
                        input_shape=input_shape,
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        activity_regularizer=activity_regularizer,
                        bias_regularizer=bias_regularizer
                    ))

                if self.indiv[self.keys.index(
                        'batchNormalization_' + str(i + 1))]:
                    cnn.add(BatchNormalization())

                if self.indiv[self.keys.index(
                        'maxPool_' + str(i + 1))]:
                    cnn.add(MaxPooling2D(
                        pool_size=self.indiv[self.keys.index(
                            'pool_size_' + str(i + 1))],
                        padding=self.indiv[self.keys.index(
                            'padding_' + str(i + 1))]))

                cnn.add(Dropout(
                    self.indiv[self.keys.index(
                        'dropout_cnn_' + str(i + 1))]))

            cnn.add(Flatten())

            cnn_mlp = concatenate([mlp.output, cnn.output])

            # Fully connected
            for i in range(sum('units_' in s for s in self.keys)):
                if self.indiv[self.keys.index(
                        'activate_' + str(i + 1))]:
                    cnn_mlp = Dense(
                        units=self.indiv[self.keys.index(
                            'units_' + str(i + 1))],
                        activation=self.indiv[self.keys.index(
                            'activation_dense')],
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        activity_regularizer=activity_regularizer
                    )(cnn_mlp)

                    cnn_mlp = Dropout(self.indiv[self.keys.index(
                        'dropout_dense_' + str(i + 1))])(cnn_mlp)

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
                loss=self._cfg.MODEL.LOSS,
                metrics=self._cfg.MODEL.METRICS)

            self._model.summary(print_fn=self._logger.info)
            return True
        except ValueError:
            return False
