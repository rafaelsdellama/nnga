from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from nnga.architectures.base_neural_network import BaseNeuralNetwork
from nnga.architectures import INICIALIZER, REGULARIZER, create_optimizer


class MLP(BaseNeuralNetwork):

    def __init__(self, cfg, logger, datasets, indiv, keys, seed):
        super().__init__(cfg, logger, datasets, indiv, keys, seed)

    def create_model_ga(self):

        if self._cfg.GA.FEATURE_SELECTION:
            input_dim = 0
            for i in range(sum(
                    'feature_selection_' in s for s in self.keys)):
                if self.indiv[self.keys.index(
                        'feature_selection_' + str(i))]:
                    input_dim += 1
        else:
            input_dim = self._datasets['TRAIN']['CSV'].n_features

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

        try:
            self._model = Sequential()

            self._model.add(Dense(
                units=self.indiv[self.keys.index('units_0')],
                activation=self.indiv[self.keys.index(
                    'activation_dense')],
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer,
                bias_regularizer=bias_regularizer,
                input_dim=input_dim
            ))

            self._model.add(Dropout(
                self.indiv[self.keys.index(
                    'dropout_dense_0')]))

            for i in range(sum('dropout_hidden_' in s for s in self.keys)):
                if self.indiv[self.keys.index('activate_' + str(i + 1))]:
                    self._model.add(Dense(
                        units=self.indiv[self.keys.index(
                            'units_' + str(i + 1))],
                        activation=self.indiv[self.keys.index(
                            'activation_dense')],
                        kernel_initializer=kernel_initializer,
                        kernel_regularizer=kernel_regularizer,
                        activity_regularizer=activity_regularizer,
                        bias_regularizer=bias_regularizer,
                    ))

                    self._model.add(Dropout(
                        self.indiv[self.keys.index(
                            'dropout_dense_' + str(i + 1))]))

            self._model.add(Dense(
                units=self._datasets['TRAIN']['CSV'].n_classes,
                activation='softmax',
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer,
                activity_regularizer=activity_regularizer,
                bias_regularizer=bias_regularizer,
            ))

            self._model.compile(
                optimizer=optimizer,
                loss=self._cfg.SOLVER.LOSS,
                metrics=self._cfg.SOLVER.METRICS)

            self._model.summary(print_fn=self._logger.info)
            return True
        except ValueError:
            return False
