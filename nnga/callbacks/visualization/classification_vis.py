import numpy as np
from sklearn.metrics import confusion_matrix
import io
import itertools
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import matplotlib

matplotlib.use("Agg")


def classification_vis(model, validation_dataset):
    # Use the model to predict the values from the test_images.
    test_pred_raw = model.predict(validation_dataset)

    lbl_encoded = validation_dataset.classes
    lbl_encoded = np.array(lbl_encoded[-len(test_pred_raw) :])

    test_pred = np.argmax(test_pred_raw, axis=1)
    test_labels = np.argmax(lbl_encoded, axis=1)

    # Calculate the confusion matrix using sklearn.metrics
    cm = confusion_matrix(test_labels, test_pred)

    return {
        "Confusion Matrix": plot_to_image(
            plot_confusion_matrix(cm, class_names=validation_dataset.labels)
        ),
        "Roc Curve": plot_to_image(
            plot_roc_curve(
                lbl_encoded=lbl_encoded,
                predict_proba=test_pred_raw,
                labels=validation_dataset.labels,
            )
        ),
    }


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(
        cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2
    )

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure


def plot_roc_curve(lbl_encoded, predict_proba, labels):
    # [labels[i] for i in lbl], predict_proba
    figure = plt.figure(figsize=(8, 8))

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(labels)):
        fpr[i], tpr[i], _ = roc_curve(lbl_encoded[:, i], predict_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(len(labels)):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"ROC curve of class {labels[i]} (area = {roc_auc[i]:.2f})",
        )
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic")
        plt.legend(loc="lower right")

    return figure


def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """

    buf = io.BytesIO()

    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format="png")

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)

    return image
