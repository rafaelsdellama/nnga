from math import ceil
import numpy as np
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from nnga.generator import GENERATOR
from nnga.utils.helper import dump_tensors
from nnga.utils.metrics import cross_validation_statistics, \
    compute_metrics
from nnga.utils.data_io import save_history, save_model, \
    save_roc_curve, save_metrics


class BaseNeuralNetwork:
    """ This class implements the Model defined by genetic algorithm indiv
        Parameters
        ----------
            cfg {yacs.config.CfgNode} -- Loaded experiment config

            logger {logging} -- Simple python logging

            datasets: ImageDataset
                dataset to be used in the train/test

            indiv: Indiv
                Indiv that determine the Model architecture

            keys: List
                List of Indiv keys
        Returns
        -------
    """

    def __init__(self, cfg, logger, datasets, indiv, keys):

        self.indiv = indiv
        self.keys = keys
        self._cfg = cfg
        self._path = cfg.OUTPUT_DIR
        self._logger = logger
        self._datasets = datasets

        self._model_history = None
        self._model = None

        self.fitting_parameters = {
            "x": None,
            "epochs": self.indiv[self.keys.index('epochs')],
            "verbose": 0,
            "callbacks": None,
            "validation_data": None,
            "shuffle": False,
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

    def get_info_dataset(self, dataset):
        """Get info from dataset
        Parameters
            ----------
                dataset: BaseDataset
                    dataset to be used in the train/test
            Returns
            -------
                -Numpy array with dataset idx
                -Numpy array with dataset idx_labels
                -List with dataset class_weights
                -List with labels
        """
        if 'CSV' in dataset:
            idx = dataset['CSV'].indexes
            idx_labels = dataset['CSV'].indexes_labels
            class_weights = dataset['CSV'].class_weights
            labels = dataset['CSV'].labels

        if 'IMG' in dataset:
            idx = dataset['IMG'].indexes
            idx_labels = dataset['IMG'].indexes_labels
            class_weights = dataset['IMG'].class_weights
            labels = dataset['IMG'].labels

        return np.array(idx), np.array(idx_labels), class_weights, labels

    def create_model_ga(self, summary=True):
        """ This method create the Model defined by genetic algorithm indiv
            Parameters
            ----------
                summary: bool
                    True: print the Model architecture
                    False: does not print the Model architecture
            Returns
            -------
                True: if the Model architecture is valid
                False: if the Model architecture is not valid
            """
        pass

    def cross_validation(self, k=5, shuffle=True, seed=0):
        """Execute cross validation using the train dataset
        Parameters
            ----------
                k: int
                    Number os cross validation folds
                shuffle: bool
                    if the data will be shuffle after each epoch
                seed: int
                    Seed to be used

            Returns
            -------
                Dict with cross validation statistic
        """
        batch_size = 2 ** self.indiv[self.keys.index('batch_size')]
        idx, idx_labels, class_weights, _ = self.get_info_dataset(
            self._datasets['TRAIN'])
        self.fitting_parameters['class_weight'] = class_weights

        skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
        skf.get_n_splits(idx, idx_labels)

        accuracy = []
        for fold, (idx_train, idx_val) in enumerate(skf.split(
                idx, idx_labels)):
            self._logger.info(f"Cross Validation - Fold {fold + 1}/{k}:")
            if self.create_model_ga(summary=False):
                self._make_generator(self._datasets['TRAIN'], idx[idx_train],
                                     self._datasets['TRAIN'], idx[idx_val],
                                     batch_size)
                self.fit()
                acc = self.evaluate()

            else:
                acc = 0

            accuracy.append(acc)
            dump_tensors()

        return cross_validation_statistics(accuracy)

    def train_test_split(self, seed=0):
        """Execute train test split using the train dataset
        Parameters
            ----------
                seed: int
                    Seed to be used to separate train test folds

            Returns
            -------
                Return the fitness (float)
        """
        if self.create_model_ga():
            batch_size = 2 ** self.indiv[self.keys.index('batch_size')]
            idx, idx_labels, class_weights, _ = self.get_info_dataset(
                self._datasets['TRAIN'])
            self.fitting_parameters['class_weight'] = class_weights

            idx_train, idx_val, _, _ = \
                train_test_split(idx, idx_labels, test_size=0.2,
                                 random_state=seed)

            self._make_generator(self._datasets['TRAIN'], idx_train,
                                 self._datasets['TRAIN'], idx_val,
                                 batch_size)

            self.fit()

            return self.evaluate()

        else:
            return 0.0

    def train(self):
        """Train the model using the train dataset and evaluate
        using the test dataset
        Parameters
            ----------

            Returns
            -------
                Return the fitness (float)
        """
        if self.create_model_ga():
            batch_size = 2 ** self.indiv[self.keys.index('batch_size')]
            idx_train, _, class_weights, _ = self.get_info_dataset(
                self._datasets['TRAIN'])
            idx_val, _, _, _ = self.get_info_dataset(self._datasets['VAL'])
            self.fitting_parameters['class_weight'] = class_weights

            self._make_generator(self._datasets['TRAIN'], idx_train,
                                 self._datasets['VAL'], idx_val,
                                 batch_size)

            self.fit()

            return self.evaluate()

        else:
            return 0.0

    def _make_generator(self, x_train, idx_train, x_val, idx_val, batch_size):
        """Make train and val generator
        Parameters
            ----------
            x_train: BaseDataset
                Dataset to be used by the train generator
            idx_train: list
                List with train sample idx to be used by the train generator
            x_val: BaseDataset
                Dataset to be used by the val generator
            idx_val: list
                List with val sample idx to be used by the val generator
            batch_size: int
                batch size

            Returns
            -------
        """
        self.fitting_parameters['x'] = \
            GENERATOR.get(self._cfg.MODEL.ARCHITECTURE)(
                x_train,
                idx_train,
                batch_size,
                self._cfg.DATASET.TRAIN_SHUFFLE,
                self._attributes,
                self.indiv[self.keys.index('scaler')],
                self._cfg.DATASET.PRESERVE_IMG_RATIO
            )

        self.fitting_parameters['steps_per_epoch'] = \
            ceil(len(idx_train) / batch_size)

        self.fitting_parameters['validation_data'] = \
            GENERATOR.get(self._cfg.MODEL.ARCHITECTURE)(
                x_val,
                idx_val,
                batch_size,
                self._cfg.DATASET.VAL_SHUFFLE,
                self._attributes,
                self.indiv[self.keys.index('scaler')],
                self._cfg.DATASET.PRESERVE_IMG_RATIO
            )

        self.fitting_parameters['validation_steps'] = \
            ceil(len(idx_val) / batch_size)

    def fit(self):
        """Fit the model"""
        self._model_history = self._model.fit(**self.fitting_parameters)

    def predict(self):
        """Predict using trained model
        Parameters
            ----------

            Returns
            -------
                lbl: class label used to predict
                lbl_encoded: class label encoded used to predict
                predict: class label predicted buy the model
                predict_encoded: predict proba predicted buy the model
        """
        self.fitting_parameters['validation_data'].reset_generator()

        predict_encoded = self._model.predict(
            x=self.fitting_parameters['validation_data'],
            steps=self.fitting_parameters['validation_steps'],
            verbose=self.fitting_parameters['verbose'],
            max_queue_size=self.fitting_parameters['max_queue_size'],
            workers=self.fitting_parameters['workers'],
            use_multiprocessing=self.fitting_parameters['use_multiprocessing']
        )

        lbl_encoded = self.fitting_parameters['validation_data'].classes
        lbl_encoded = lbl_encoded[
                      -self.fitting_parameters['validation_data'].num_indexes:]
        predict_encoded = predict_encoded[
                  -self.fitting_parameters['validation_data'].num_indexes:]

        lbl = [np.argmax(t) for t in lbl_encoded]
        predict = [np.argmax(t) for t in predict_encoded]

        return lbl, lbl_encoded, predict, predict_encoded

    def evaluate(self):
        """Evaluate the model using predict function
        Parameters
            ----------

            Returns
            -------
                Acc from the model
        """
        lbl, _, predict, _ = self.predict()

        acc = balanced_accuracy_score(lbl, predict)
        confusion_m = confusion_matrix(lbl, predict)

        self._logger.info(f"balanced accuracy: {acc}")
        self._logger.info(f"confusion matrix: \n{confusion_m}")

        return acc

    def compute_metrics(self):
        """Compute all metrics
        Parameters
            ----------

            Returns
            -------
                Dict containing all metric results
        """
        lbl, _, predict, predict_encoded = self.predict()
        idx, _, _, labels = self.get_info_dataset(self._datasets['VAL'])

        return compute_metrics(lbl, predict,
                               predict_encoded, labels, idx)

    def save_metrics(self, seed):
        """Save model metrics
        Parameters
            ----------
                seed: int
                    Seed used from GA

            Returns
            -------
        """
        metrics = self.compute_metrics()
        save_metrics(self._path, seed, metrics)

    def save_roc_curve(self, seed):
        """Save model Roc Curve
        Parameters
            ----------
                seed: int
                    Seed used from GA

            Returns
            -------
        """
        lbl, _, _, predict_proba = self.predict()
        _, _, _, labels = self.get_info_dataset(self._datasets['TRAIN'])
        save_roc_curve(self._path, seed, lbl, predict_proba, labels)

    def summary(self):
        """Print the model summary"""
        self._model.summary(print_fn=self._logger.info)

    def save_model(self, seed):
        """Save model
        Parameters
            ----------
                seed: int
                    Seed used from GA

            Returns
            -------
        """
        save_model(self._path, seed, self._model)

    def save_history(self, seed):
        """Save model history
        Parameters
            ----------
                seed: int
                    Seed used from GA

            Returns
            -------
        """
        save_history(self._path, seed, self._model_history.history)
