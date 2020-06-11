from tensorflow.keras.callbacks import Callback
import os
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2, array_ops
from tensorflow.python.keras import backend as K
from nnga.callbacks.visualization import make_custom_visualization_by_task


class TensorBoard(Callback):
    """Callback to create visualization  on TensorBoard.
    Arguments:
        log_dir: str
            The path of the directory where to save the log files to be
            parsed by TensorBoard.
        task: model task
            tasks: [Classification, Segmentation]
        histogram_freq: int
            Frequency (in epochs) at which to compute activation and
            weight histograms for the layers of the model. If set to 0,
            histograms won't be computed. Validation data (or split) must
            be specified for histogram visualizations.
        write_graph: bool
            Whether to visualize the graph in TensorBoard. The log file
            can become quite large when write_graph is set to True.
        write_images: bool
            Whether to write model weights to visualize as image in
            TensorBoard.
        update_freq: `'batch'` or `'epoch'` or integer.
            When using `'batch'`, writes the losses and metrics to
            TensorBoard after each batch.
            The same applies for `'epoch'`. If using an integer, let's
            say `1000`, the callback will write the metrics and losses to
            TensorBoard every 1000 batches. Note that writing too frequently
            to TensorBoard can slow down your training.
        val_dataset: dataset
            validation dataset for custom visualization
    """

    def __init__(
        self,
        log_dir=".",
        task=None,
        histogram_freq=0,
        write_graph=True,
        write_images=False,
        update_freq="epoch",
        val_dataset=None,
    ):
        super(TensorBoard, self).__init__()
        self.log_dir = os.path.join(log_dir)
        self._train_writer = tf.summary.create_file_writer(
            os.path.join(self.log_dir, "train")
        )
        self._val_writer = tf.summary.create_file_writer(
            os.path.join(self.log_dir, "val")
        )
        self._graph_writer = tf.summary.create_file_writer(
            os.path.join(self.log_dir, "graph")
        )
        self.task = task
        self.histogram_freq = histogram_freq
        self.write_graph = write_graph
        self.write_images = write_images
        self.update_freq = 1 if update_freq == "batch" else update_freq

        # Lazily initialized in order to avoid creating event files when
        # not needed.
        self._writers = {}

        self._val_dataset = val_dataset
        self._custom_vis = make_custom_visualization_by_task(self.task)

    def on_train_begin(self, logs=None):
        if self.write_graph:
            self._write_graph()

    def on_epoch_end(self, epoch, logs=None):
        """Runs metrics and histogram summaries at epoch end."""
        self._log_epoch_metrics(epoch, logs)

        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_weights(epoch)

    def on_train_end(self, logs=None):
        self._close_writers()

    def _write_graph(self):
        with self._graph_writer.as_default():
            with summary_ops_v2.always_record_summaries():
                if not self.model.run_eagerly:
                    summary_ops_v2.graph(K.get_graph(), step=0)
                self._graph_writer.flush()

    def _close_writers(self):
        self._train_writer.close()
        self._val_writer.close()
        self._graph_writer.close()

    def _log_epoch_metrics(self, epoch, logs):
        """Writes epoch metrics out as scalar summaries.
        Arguments:
            epoch: Int. The global step to use for TensorBoard.
            logs: Dict. Keys are scalar summary names, values are scalars.
        """
        if not logs:
            return

        train_logs = {
            k: v for k, v in logs.items() if not k.startswith("val_")
        }
        val_logs = {k: v for k, v in logs.items() if k.startswith("val_")}

        with summary_ops_v2.always_record_summaries():
            if train_logs:
                with self._train_writer.as_default():
                    for name, value in train_logs.items():
                        summary_ops_v2.scalar(name, value, step=epoch)

                    summary_ops_v2.scalar(
                        "learning_rate", self.model.optimizer.lr, step=epoch
                    )

            if val_logs:
                with self._val_writer.as_default():
                    for name, value in val_logs.items():
                        name = name[4:]  # Remove 'val_' prefix.
                        summary_ops_v2.scalar(name, value, step=epoch)

                    if self._val_dataset is not None and \
                            self._custom_vis is not None:
                        for name, img in self._custom_vis(
                            self.model, self._val_dataset
                        ).items():
                            summary_ops_v2.image(
                                name, img, step=epoch,
                            )

    def _log_weights(self, epoch):
        """Logs the weights of the Model to TensorBoard."""
        with self._train_writer.as_default():
            with summary_ops_v2.always_record_summaries():
                for layer in self.model.layers:
                    for weight in layer.weights:
                        weight_name = weight.name.replace(":", "_")
                        summary_ops_v2.histogram(
                            weight_name, weight, step=epoch
                        )
                        if self.write_images:
                            self._log_weight_as_image(
                                weight, weight_name, epoch
                            )
                self._train_writer.flush()

    def _log_weight_as_image(self, weight, weight_name, epoch):
        """Logs a weight as a TensorBoard image."""
        w_img = array_ops.squeeze(weight)
        shape = K.int_shape(w_img)
        if len(shape) == 1:  # Bias case
            w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
        elif len(shape) == 2:  # Dense layer kernel case
            if shape[0] > shape[1]:
                w_img = array_ops.transpose(w_img)
                shape = K.int_shape(w_img)
            w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
        elif len(shape) == 3:  # ConvNet case
            if K.image_data_format() == "channels_last":
                # Switch to channels_first to display every kernel
                # as a separate image.
                w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
                shape = K.int_shape(w_img)
            w_img = array_ops.reshape(w_img, [shape[0], shape[1],
                                              shape[2], 1])

        shape = K.int_shape(w_img)
        # Not possible to handle 3D convnets etc.
        if len(shape) == 4 and shape[-1] in [1, 3, 4]:
            summary_ops_v2.image(weight_name, w_img, step=epoch)
