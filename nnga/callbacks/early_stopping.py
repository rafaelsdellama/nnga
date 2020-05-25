import numpy as np
import warnings
from tensorflow.keras.callbacks import Callback
from nnga.utils.logger import get_logger


class EarlyStopping(Callback):
    """Stop training when the monitor stops to increasing/decreasing.

    Arguments:
         monitor: str
            Quantity to be monitored. (default: {loss})
        min_delta: float
            Minimum change in the monitored quantity to qualify as an improvement,
            i.e. an absolute change of less than min_delta, will count as no improvement.
        patience: int
            Number of epochs to wait after min has been hit. After this
            number of no improvement, training stops. (default: {0})
        verbose: int
            0: quiet, 1: update messages. (default: {0})
        mode: str
            One of {"auto", "min", "max"}.
            In min mode, training will stop when the quantity monitored has stopped decreasing;
            in max mode it will stop when the quantity monitored has stopped increasing;
            in auto mode, the direction is automatically inferred from the name of the monitored quantity.
        baseline: Baseline value for the monitored quantity.
            Training will stop if the model doesn't show improvement over the
            baseline.
        restore_best_weights: Whether to restore model weights from
            the epoch with the best value of the monitored quantity.
            If False, the model weights obtained at the last step of
            training are used.
    """

    def __init__(self, monitor='val_loss',
                 min_delta=0.0,
                 patience=0,
                 verbose=0,
                 mode='auto',
                 baseline=None,
                 restore_best_weights=False):
        super(EarlyStopping, self).__init__()

        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None
        self.logger = get_logger("NNGA")

        if self.mode not in ["auto", "min", "max"]:
            warnings.warn(
                f"EarlyStopping mode {self.mode} is unknown, "
                "fallback to auto mode.",
                RuntimeWarning,
            )
            self.mode = "auto"

        if self.mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        elif self.mode == 'max':
            self.monitor_op = np.greater
            self.min_delta *= 1
        else:
            if 'acc' in self.monitor:
                self.mode = 'max'
                self.monitor_op = np.greater
                self.min_delta *= 1
            else:
                self.mode = 'min'
                self.monitor_op = np.less
                self.min_delta *= -1

            if self.verbose:
                self.logger.info(f"EarlyStopping mode is automatically inferred to '{self.mode}'")

        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        if self.baseline is not None:
            self.best = self.baseline
        else:
            self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)

        if current is None:
            return

        if self.monitor_op(current - self.min_delta, self.best):
            self.best = current
            self.wait = 0
            # Record the best weights if current results is better (less).
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose:
                        self.logger.info('Restoring model weights from the end of the best epoch.')
                    self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0 and self.verbose:
            self.logger.info(f'Epoch {self.stopped_epoch + 1}: early stopping')
