from tensorflow.keras.callbacks import Callback
import numpy as np
import os
import warnings
from nnga.utils.logger import get_logger
from nnga.utils.data_io import (
    save_model, save_cfg, load_model,
    save_train_state, load_train_state
)


class ModelCheckpoint(Callback):
    """Callback to save the Keras model or model weights at some frequency.

    Arguments:
        filepath: str
            Path to save the model file.
        monitor: str
            Quantity to be monitored. (default: {loss})
        verbose: int
            0: quiet, 1: update messages. (default: {0})
        mode: str
            One of {"auto", "min", "max"}.
            In min mode, training will stop when the quantity monitored has stopped decreasing;
            in max mode it will stop when the quantity monitored has stopped increasing;
            in auto mode, the direction is automatically inferred from the name of the monitored quantity.
        save_best_only: bool
            If save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
            If filepath doesn't contain formatting options like {epoch} then filepath will be overwritten by each
            new better model.
    """

    def __init__(
        self,
        filepath,
        monitor='val_loss',
        verbose=0,
        save_best_only=False,
        mode='auto',
    ):
        super(ModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.modelpath = os.path.join(filepath, 'model')
        self.monitor = monitor
        self.verbose = verbose
        self.mode = mode
        self.save_best_only = save_best_only
        self.logger = get_logger("NNGA")

        if os.path.exists(os.path.join(self.modelpath, 'train_state.json')):
            self.train_state = load_train_state(self.modelpath)
        else:
            self.train_state = {}

        if self.mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"ModelCheckpoint mode {self.mode} is unknown, "
                "fallback to auto mode.",
                RuntimeWarning,
            )
            self.mode = "auto"

        if self.mode == 'min':
            self.monitor_op = np.less
            self.best = self.train_state.get("best_score", np.Inf)
        elif self.mode == 'max':
            self.monitor_op = np.greater
            self.best = self.train_state.get("best_score", -np.Inf)
        else:
            if 'acc' in self.monitor:
                self.mode = 'max'
                self.monitor_op = np.greater
                self.best = self.train_state.get("best_score", -np.Inf)
            else:
                self.mode = 'min'
                self.monitor_op = np.less
                self.best = self.train_state.get("best_score", np.Inf)

            if self.verbose:
                self.logger.info(f"ModelCheckpoint mode is automatically inferred to '{self.mode}'")

    def on_train_begin(self, logs=None):
        save_cfg(self.filepath)

        if os.path.exists(os.path.join(self.modelpath, 'saved_model.pb')):
            self.model = load_model(self.modelpath)
            if self.verbose:
                self.logger.info(f"Weights loaded from file found in {self.modelpath}")

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)

        if self.save_best_only:
            if current is None:
                warnings.warn(f'Can save best model only with {self.monitor} available, '
                              f'skipping.', RuntimeWarning)

            elif self.monitor_op(current, self.best):
                self.best = current
                self.save_checkpoint(epoch)
                if self.verbose:
                    self.logger.info(f'\nEpoch {epoch + 1}: {self.monitor} improved from {self.best} to {current},'
                                     f' saving model to {self.modelpath}')
            else:
                if self.verbose:
                    self.logger.info(f'\nEpoch {epoch + 1}: {self.monitor} did not improve from {self.best}')
        else:
            self.best = current
            self.save_checkpoint(epoch)
            if self.verbose:
                self.logger.info(f"\nEpoch {epoch + 1}: saving model to {self.modelpath}")

    def save_checkpoint(self, epoch):
        save_model(self.modelpath, self.model)
        self.train_state["epoch"] = epoch
        self.train_state["best_score"] = self.best
        save_train_state(self.modelpath, self.train_state)
