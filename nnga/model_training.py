import numpy as np
import math
import copy
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from nnga.utils.metrics import cross_validation_statistics, compute_metrics
from nnga.utils.data_io import (
    save_history,
    save_roc_curve,
    save_metrics,
    load_train_state,
)
from nnga.architectures import OPTIMIZERS
from nnga.callbacks.cosine_decay_with_warmup import WarmUpCosineDecayScheduler
from nnga.callbacks.early_stopping import EarlyStopping
from nnga.callbacks.model_checkpoint import ModelCheckpoint
from nnga.callbacks.tensor_board import TensorBoard
from nnga.architectures.segmentation.losses import dice_loss
from nnga.architectures.segmentation.metrics import iou_coef


class ModelTraining:
    """ This class implements the Model defined by genetic algorithm indiv
        Parameters
        ----------
            cfg yacs.config.CfgNode}
                Loaded experiment config

            model: BaseModel
                Model

            logger logging
                Simple python logging

            datasets: BaseDataset
                dataset to be used in the train/test

            indiv: Indiv
                Indiv that determine the Model architecture

            keys: List
                List of Indiv keys

            features_selected: list
                list of feature selected index
        Returns
        -------
    """

    def __init__(
        self,
        cfg,
        model,
        logger,
        datasets,
        indiv=None,
        keys=None,
        features_selected=None,
    ):

        self.indiv = indiv
        self.keys = keys
        self._cfg = cfg
        self._logger = logger
        self._datasets = datasets
        self._model = model.get_model()

        self.cv = self._cfg.SOLVER.CROSS_VALIDATION
        self.cv_folds = self._cfg.SOLVER.CROSS_VALIDATION_FOLDS
        self.test_size = self._cfg.SOLVER.TEST_SIZE

        if not isinstance(self.cv_folds, int) or self.cv_folds <= 1:
            raise ValueError(
                "SOLVER.CROSS_VALIDATION_FOLDS must be a int "
                "number (1 < SOLVER.CROSS_VALIDATION_FOLDS)"
            )

        if (
            not isinstance(self.test_size, float)
            or self.test_size <= 0
            or self.test_size >= 1
        ):
            raise ValueError(
                "SOLVER.TEST_SIZE must be a float "
                "number (0 < SOLVER.TEST_SIZE < 1)"
            )

        if not isinstance(self.cv, bool):
            raise ValueError("SOLVER.CROSS_VALIDATION must be a bool")

        self._datasets["TRAIN"].features_selected = features_selected
        self._datasets["VAL"].features_selected = features_selected

        if os.path.exists(
            os.path.join(self._cfg.OUTPUT_DIR, "model", "train_state.json")
        ):
            self.train_state = load_train_state(
                os.path.join(self._cfg.OUTPUT_DIR, "model")
            )
        else:
            self.train_state = {}

        loss, metrics = get_losses_and_metrics(self._cfg)
        self.compile_parameters = {
            "loss": loss,
            "metrics": metrics,
        }
        self.fitting_parameters = {
            "x": self._datasets["TRAIN"],
            "epochs": self._cfg.SOLVER.EPOCHS,
            "validation_data": self._datasets["VAL"],
            "class_weight": self._datasets["TRAIN"].class_weights,
            "steps_per_epoch": len(self._datasets["TRAIN"]),
            "validation_steps": len(self._datasets["VAL"]),
            "verbose": 1,
            "callbacks": None,
            "shuffle": False,
            "validation_freq": 1,
            "max_queue_size": 10,
            "workers": 0,
            "use_multiprocessing": True,
            "initial_epoch": self.train_state.get("epoch", 0),
        }

        self._path = cfg.OUTPUT_DIR
        self._labels = datasets["TRAIN"].labels
        self._model_history = None

        if self._cfg.MODEL.BACKBONE == "GASearch":
            self._configure_compiler_ga()
            self._configure_fit_ga()
            self._logger.info(f"Model Trainner from GA created!")
        else:
            self._configure_compiler()
            self._configure_fit()
            self._logger.info(f"Model Trainner created!")

        self._compile()
        self._logger.info(f"Model compiled!")

    def _configure_compiler_ga(self):
        """
        Set compile parameters from GA indiv
        """
        self.compile_parameters.update(
            {
                "optimizer": OPTIMIZERS.get(
                    self.indiv[self.keys.index("optimizer")]
                )(self.indiv[self.keys.index("learning_rate")]),
            }
        )

    def _configure_compiler(self):
        """
        Set compile parameters from config
        """
        self.compile_parameters.update(
            {
                "optimizer": OPTIMIZERS.get(self._cfg.SOLVER.OPTIMIZER)(
                    self._cfg.SOLVER.BASE_LEARNING_RATE
                ),
            }
        )

    def _configure_fit_ga(self):
        """ Set fit parameters from GA indiv """
        self.fitting_parameters.update(
            {
                "epochs": self.indiv[self.keys.index("epochs")],
                "callbacks": None,
                "initial_epoch": 0,
            }
        )

    def _configure_fit(self):
        """ Set fit parameters from config """
        self.fitting_parameters.update(
            {
                "epochs": self._cfg.SOLVER.EPOCHS,
                "callbacks": self._make_callbacks(
                    val_dataset=self._datasets["VAL"]
                ),
                "initial_epoch": self.train_state.get("epoch", 0),
            }
        )

    def _compile(self):
        """ Compile the model """
        self._model.compile(**self.compile_parameters)

    def fit(self):
        """Fit the model"""
        self._model_history = self._model.fit(**self.fitting_parameters)

    def train_test_split(self, random_state=0):
        """Execute train test split using the train dataset
        Parameters
            ----------
                random_state: int
                    random_state to be used to separate train test folds

            Returns
            -------
                loss value & metrics values
        """
        Wsave = self._model.get_weights()
        fitting_parameters = self.fitting_parameters.copy()

        idx, idx_labels = self._datasets["TRAIN"].indexes

        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=self.test_size, random_state=random_state
        )
        sss.get_n_splits(idx, idx_labels)
        idx_train, idx_val = next(sss.split(idx, idx_labels))

        train_dataset = copy.deepcopy(self._datasets["TRAIN"])
        #train_dataset.set_index(idx_train)
        train_dataset.set_index(idx_train[0:len(idx_val)//4])
        test_dataset = copy.deepcopy(self._datasets["TRAIN"])
        #test_dataset.set_index(idx_val)
        test_dataset.set_index(idx_val[0:len(idx_val)//4])

        self.fitting_parameters.update(
            {
                "x": train_dataset,
                "validation_data": test_dataset,
                "steps_per_epoch": len(train_dataset),
                "validation_steps": len(test_dataset),
                "callbacks": None,
                "initial_epoch": 0,
            }
        )

        self.fit()
        metrics = self.compute_metrics()

        self._model.set_weights(Wsave)
        self.fitting_parameters.update(fitting_parameters)

        return metrics

    def cross_validation(self, shuffle=True, random_state=0, save=False):
        """Execute cross validation using the train dataset
        Parameters
            ----------
                shuffle: bool
                    if the data will be shuffle after each epoch
                random_state: int
                    random_state for StratifiedKFold
                save: bool
                    if true, save the metrics result

            Returns
            -------
                Pandas DataFrame with cross validation results
        """
        Wsave = self._model.get_weights()
        fitting_parameters = self.fitting_parameters.copy()

        idx, idx_labels = self._datasets["TRAIN"].indexes

        skf = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=shuffle, random_state=random_state
        )
        skf.get_n_splits(idx, idx_labels)

        evaluate_results = []
        for fold, (idx_train, idx_val) in enumerate(
            skf.split(idx, idx_labels)
        ):
            self._logger.info(
                f"Cross Validation - Fold {fold + 1}/{self.cv_folds}:"
            )

            train_dataset = copy.deepcopy(self._datasets["TRAIN"])
            train_dataset.set_index(idx_train)
            test_dataset = copy.deepcopy(self._datasets["TRAIN"])
            test_dataset.set_index(idx_val)

            self.fitting_parameters.update(
                {
                    "x": train_dataset,
                    "validation_data": test_dataset,
                    "steps_per_epoch": len(train_dataset),
                    "validation_steps": len(test_dataset),
                    "callbacks": None,
                    "initial_epoch": 0,
                }
            )

            self.fit()
            evaluate = self.evaluate()
            evaluate_results.append(evaluate)

            metrics = self.compute_metrics()
            self._logger.info(
                f"balanced accuracy: {metrics['balanced_accuracy_score']}"
            )
            self._logger.info(
                f"confusion matrix: \n{metrics['confusion_matrix']}"
            )

            self._model.set_weights(Wsave)
            self.fitting_parameters.update(fitting_parameters)

        cv = cross_validation_statistics(evaluate_results)
        self._logger.info(f"Cross validation statistics:\n{cv}")

        if save:
            save_metrics(
                self._path, {"cross_validation": cv.to_string(index=True)}
            )
        return cv

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
        self.fitting_parameters["validation_data"].reset_generator()

        predict_encoded = self._model.predict(
            x=self.fitting_parameters["validation_data"],
            steps=self.fitting_parameters["validation_steps"],
            verbose=self.fitting_parameters["verbose"],
            max_queue_size=self.fitting_parameters["max_queue_size"],
            workers=self.fitting_parameters["workers"],
            use_multiprocessing=self.fitting_parameters["use_multiprocessing"],
        )

        lbl_encoded = self.fitting_parameters["validation_data"].classes
        lbl_encoded = lbl_encoded[-len(predict_encoded) :]

        lbl = [np.argmax(t) for t in lbl_encoded]
        predict = [np.argmax(t) for t in predict_encoded]

        return lbl, lbl_encoded, predict, predict_encoded

    def evaluate(self):
        """Evaluate the model using evaluate function
        Parameters
            ----------

            Returns
            -------
                 loss value & metrics values
        """
        evaluate = self._model.evaluate(
            x=self.fitting_parameters["validation_data"],
            steps=self.fitting_parameters["validation_steps"],
            verbose=self.fitting_parameters["verbose"],
            max_queue_size=self.fitting_parameters["max_queue_size"],
            workers=self.fitting_parameters["workers"],
            use_multiprocessing=self.fitting_parameters["use_multiprocessing"],
        )

        if math.isnan(evaluate[0]):
            evaluate[0] = float("inf")
        return evaluate

    def compute_metrics(self, save=False):
        """Compute all metrics

        IF save is True, save the metrics, Roc Curve and
        model history

        Parameters
            ----------

            save: bool
                if true, save the metrics result

            Returns
            -------
                Dict containing all metric results
        """
        lbl, _, predict, predict_encoded = self.predict()

        metrics = compute_metrics(
            lbl,
            predict,
            predict_encoded,
            self._labels,
            self.fitting_parameters["validation_data"].indexes[0],
        )
        if save:
            save_metrics(self._path, metrics)
            save_roc_curve(self._path, lbl, predict_encoded, self._labels)
            save_history(self._path, self._model_history.history)

        return metrics

    def _make_callbacks(self, val_dataset=None):
        total_steps = int(
            len(self._datasets["TRAIN"]) * self._cfg.SOLVER.EPOCHS
        )
        callbacks = [
            WarmUpCosineDecayScheduler(
                learning_rate_base=self._cfg.SOLVER.BASE_LEARNING_RATE,
                total_steps=total_steps,
                warmup_steps=total_steps // 10,
                hold_base_rate_steps=total_steps // 3,
                verbose=self._cfg.VERBOSE,
            ),
            EarlyStopping(
                patience=10,
                verbose=self._cfg.VERBOSE,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                filepath=self._cfg.OUTPUT_DIR,
                verbose=self._cfg.VERBOSE,
                save_best_only=True,
            ),
            TensorBoard(
                log_dir=self._cfg.OUTPUT_DIR,
                task=self._cfg.TASK,
                val_dataset=val_dataset,
            ),
        ]
        return callbacks


def get_losses_and_metrics(cfg):
    """Return loss and metrics functions for specific task"""
    options = {
        "Classification": {
            "loss": cfg.SOLVER.LOSS,
            "metrics": cfg.SOLVER.METRICS,
        },
        "Segmentation": {
            "loss": dice_loss,
            "metrics": [iou_coef],
        }
    }.get(cfg.TASK)
    return options["loss"], options["metrics"]
