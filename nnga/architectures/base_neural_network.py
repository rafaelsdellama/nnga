from sklearn.model_selection import train_test_split
from nnga.generator import GENERATOR
from math import ceil
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score


class BaseNeuralNetwork:

    def __init__(self, cfg, logger, datasets, indiv, keys, seed):

        self.indiv = indiv
        self.keys = keys
        self._cfg = cfg
        self._path = cfg.OUTPUT_DIR
        self._logger = logger
        self._datasets = datasets
        self._seed = seed

        self._model_history = None
        self._model = None

        self.fitting_parameters = {
            "x": None,
            "epochs": self.indiv[self.keys.index('epochs')],
            "verbose": 0,
            "callbacks": None,
            "validation_data": None,
            "shuffle": True,
            "class_weight": None,
            "steps_per_epoch": None,
            "validation_steps": None,
            "validation_freq": 1,
            "max_queue_size": 10,
            "workers": 0,
            "use_multiprocessing": True,
        }

        if cfg.GA.FEATURE_SELECTION:
            indexes = [i for i, value in enumerate(self.keys)
                       if 'feature_selection_' in value]
            self._attributes = [i for i, value in enumerate(indexes)
                                if self.indiv[value]]
        else:
            self._attributes = None

        self._prepare_data()

    def create_model_ga(self):
        pass

    def _prepare_data(self):
        batch_size = 2 ** self.indiv[self.keys.index('batch_size')]
        if 'CSV' in self._datasets['TRAIN']:
            idx = self._datasets['TRAIN']['CSV'].indexes
            idx_labels = self._datasets['TRAIN']['CSV'].indexes_labels
            self.fitting_parameters['class_weight'] = \
                self._datasets['TRAIN']['CSV'].class_weights

        if 'IMG' in self._datasets['TRAIN']:
            idx = self._datasets['TRAIN']['IMG'].indexes
            idx_labels = self._datasets['TRAIN']['IMG'].indexes_labels
            self.fitting_parameters['class_weight'] = \
                self._datasets['TRAIN']['IMG'].class_weights

        x_train, x_val, _, _ = \
            train_test_split(idx, idx_labels, test_size=0.2,
                             random_state=self._seed)

        self._make_generator(x_train, x_val, batch_size)

    def _make_generator(self, x_train, x_val, batch_size):
        self.fitting_parameters['x'] = GENERATOR.get(
            self._cfg.MODEL.ARCHITECTURE)(
            self._datasets['TRAIN'],
            x_train,
            batch_size,
            self._cfg.DATASET.TRAIN_SHUFFLE,
            self._attributes,
            self.indiv[self.keys.index('scaler')],
            self._cfg.DATASET.PRESERVE_IMG_RATIO)
        self.fitting_parameters['steps_per_epoch'] = \
            ceil(len(x_train) / batch_size)

        self.fitting_parameters['validation_data'] = GENERATOR.get(
            self._cfg.MODEL.ARCHITECTURE)(
            self._datasets['TRAIN'],
            x_val,
            batch_size,
            self._cfg.DATASET.VAL_SHUFFLE,
            self._attributes,
            self.indiv[self.keys.index('scaler')],
            self._cfg.DATASET.PRESERVE_IMG_RATIO)

        self.fitting_parameters['validation_steps'] = \
            ceil(len(x_val) / batch_size)

    def fit(self):
        self._model.fit(**self.fitting_parameters)

    def evaluate(self):
        self.fitting_parameters['validation_data'].reset_generator()

        predict = self._model.predict(
            x=self.fitting_parameters['validation_data'],
            steps=self.fitting_parameters['validation_steps'],
            verbose=self.fitting_parameters['verbose'],
            max_queue_size=self.fitting_parameters['max_queue_size'],
            workers=self.fitting_parameters['workers'],
            use_multiprocessing=self.fitting_parameters['use_multiprocessing']
        )

        lbl = [np.argmax(t) for t in
               self.fitting_parameters['validation_data'].classes]
        predict = [np.argmax(t) for t in predict]

        lbl = lbl[0:self.fitting_parameters['validation_data'].num_indexes]
        predict = predict[0:self.fitting_parameters['validation_data'].num_indexes]

        acc = balanced_accuracy_score(lbl, predict)
        confusion_m = confusion_matrix(lbl, predict)

        self._logger.info(f"confusion matrix: \n{confusion_m}")

        return acc
