import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import cv2
import scikitplot as skplt
import csv


def create_dir(path):
    """
        Create target Directory if don't exist

        Parameters
        ----------
        path : str
            Directory path

        Returns
        -------
    """
    if not os.path.exists(path):
        os.makedirs(path)


def load_statistic(path, seed):
    """
        Load statistic from GA to continue the execution

        Parameters
        ----------
        path : str
            Directory path

        seed: int
            Seed from GA

        Returns
        -------
            Pandas DataFrame with the statistic data
    """
    statistic_dir = Path(path, str(seed), f"results.csv").as_posix()
    if not os.path.exists(statistic_dir):
        raise FileNotFoundError(f"{statistic_dir} does not exist")

    return pd.read_csv(statistic_dir)


def load_pop(path, seed):
    """
        Load pop from GA to continue the execution

        Parameters
        ----------
        path : str
            Directory path

        seed: int
            Seed from GA

        Returns
        -------
            Pandas DataFrame with the pop data
    """
    pop_dir = Path(path, str(seed), f"pop.csv").as_posix()
    if not os.path.exists(pop_dir):
        raise FileNotFoundError(f"{pop_dir} does not exist")
    return pd.read_csv(pop_dir)


def save_statistic(path, seed, statistic):
    """
        Save statistic from GA

        Parameters
        ----------
        path : str
            Directory path

        seed: int
            Seed from GA

        statistic: DataFrame
            List of dict

        Returns
        -------
    """
    create_dir(Path(path, str(seed)))

    statistic_dir = Path(path, str(seed), f"results.csv").as_posix()
    plot_dir = Path(path, str(seed), f"Generations_GA.png").as_posix()

    df = pd.DataFrame(
        statistic, columns=["mean_fitness", "best_fitness", "best_indiv"]
    )
    df.to_csv(statistic_dir, index=False, header=True)

    plt.plot(list(df["mean_fitness"]), "r")
    plt.plot(list(df["best_fitness"]), "g")
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Generations GA")
    plt.legend(["Mean Fitness", "Best Fitness"])

    plt.savefig(plot_dir)
    plt.clf()  # Clear the current figure.
    plt.cla()  # Clear the current axes.
    plt.close()


def save_pop(path, seed, pop):
    """
        Save pop from GA

        Parameters
        ----------
        path : str
            Directory path

        seed: int
            Seed from GA

        pop: DataFrame
            Pandas DataFrame with the pop data

        Returns
        -------
    """
    create_dir(Path(path, str(seed)))
    pop_dir = Path(path, str(seed), f"pop.csv").as_posix()

    pop_list = []
    fitness_list = []

    for i in range(len(pop.indivs)):
        pop_list.append(pop.indivs[i].chromosome)
        fitness_list.append(pop.indivs[i].fitness)

    last_pop = pd.DataFrame({"indiv": pop_list, "fitness": fitness_list})
    last_pop.to_csv(pop_dir, index=False, header=True)


def save_history(path, seed, model_history):
    """
        Save model history from model

        Parameters
        ----------
        path : str
            Directory path

        seed: int
            Seed from GA

        model_history: model_history
            model history

        Returns
        -------
    """
    create_dir(Path(path, str(seed)))
    legend = []
    plot_dir = Path(path, str(seed), f"history_model.png").as_posix()
    if "acc" in model_history:
        plt.plot(model_history["acc"], "r")
        legend.append("Training accuracy")
    elif "accuracy" in model_history:
        plt.plot(model_history["accuracy"], "r")
        legend.append("Training accuracy")
    elif "categorical_accuracy" in model_history:
        plt.plot(model_history["categorical_accuracy"], "r")
        legend.append("Training accuracy")

    if "val_acc" in model_history:
        plt.plot(model_history["val_acc"], "g")
        legend.append("Validation accuracy")
    elif "val_accuracy" in model_history:
        plt.plot(model_history["val_accuracy"], "g")
        legend.append("Validation accuracy")
    elif "val_categorical_accuracy" in model_history:
        plt.plot(model_history["val_categorical_accuracy"], "g")
        legend.append("Validation accuracy")

    if "loss" in model_history:
        plt.plot(model_history["loss"], "b", linestyle="--")
        legend.append("Training loss")
    if "val_loss" in model_history:
        plt.plot(model_history["val_loss"], "y", linestyle="--")
        legend.append("Validation loss")

    plt.xticks(
        np.arange(
            0,
            len(model_history["loss"]) / 10
            if len(model_history["loss"]) > 10
            else 5,
            len(model_history["loss"]) / 10
            if len(model_history["loss"]) > 10
            else 1,
        )
    )
    plt.xlabel("Num of Epochs")
    plt.ylabel("Accuracy / Loss")
    plt.title("Training vs Validation")
    plt.legend(legend)

    plt.savefig(plot_dir)
    plt.clf()
    plt.cla()
    plt.close()


def save_model(path, seed, model):
    """
        Save model

        Parameters
        ----------
        path : str
            Directory path

        seed: int
            Seed from GA

        model: model
            model to be saved

        Returns
        -------
    """
    create_dir(Path(path, str(seed)))

    model_dir = Path(path, str(seed), f"model").as_posix()
    with open(model_dir + ".json", "w") as json_file:
        json_file.write(model.to_json())
    model.save_weights(model_dir + ".h5")


def save_roc_curve(path, seed, lbl, predict_proba, labels):
    """
        Save Roc Curve

        Parameters
        ----------
        path : str
            Directory path

        seed: int
            Seed from GA

        lbl: numoy array
            numpy array with target class

        predict_proba: numoy array
            numpy array with predict proba class

        labels: list
            list with all target label

        Returns
        -------
    """
    create_dir(Path(path, str(seed)))

    plot_dir = Path(path, str(seed), f"roc_curve.png").as_posix()
    skplt.metrics.plot_roc([labels[i] for i in lbl], predict_proba)
    plt.savefig(plot_dir)
    plt.clf()
    plt.cla()
    plt.close()


def save_metrics(path, seed, metrics):
    """
        Save all Metrics

        Parameters
        ----------
        path : str
            Directory path

        seed: int
            Seed from GA

        metrics: Dict
            Dict containing all metrics to be saved

        Returns
        -------
    """
    create_dir(Path(path, str(seed)))
    file_name = Path(path, str(seed), f"metrics.txt").as_posix()

    file = open(file_name, "w")

    for key, value in metrics.items():
        file.write(key)
        file.write("\n")
        file.write(value)
        file.write("\n\n")

    file.close()


def load_csv_file(path, usecols, chunksize=None):
    """
        Load csv file

        Parameters
        ----------
        path : str
            CSV directory path

        usecols: List
            List with the column names to be read

        chunksize: int
            To be used in pd.read_csv

        Returns
        -------
        If chunksize is None, return Pandas DataFrame
        containing: labels | data(s) | classes
        Else return TextFileReader object for iteration

    """
    if not isinstance(path, str):
        raise TypeError("path is not string type")
    if not os.path.exists(path):
        raise FileNotFoundError("File not found with provided path.")

    with open(path, "r") as csvfile:
        try:
            dialect = csv.Sniffer().sniff(
                csvfile.read(1024 * 100), delimiters=[",", ";"]
            )

            if dialect.delimiter == ",":
                df = pd.read_csv(path, usecols=usecols, chunksize=chunksize)
            elif dialect.delimiter == ";":
                df = pd.read_csv(
                    path,
                    usecols=usecols,
                    sep=";",
                    decimal=",",
                    chunksize=chunksize,
                )

        except csv.Error:
            # Could not conclude the delimiter, defaulting to comma
            df = pd.read_csv(path, usecols=usecols, chunksize=chunksize)

    return df


def load_image(path, input_shape, preserve_ratio=True):
    """
        Load image

        Parameters
        ----------
        path : str
            Directory path

        input_shape: tuple
            Tuple of image shape

        preserve_ratio: Bool
            if True, preserve image ratio,
            else Does not preserve the image ratio

        Returns
        -------
        numpy array image

    """

    if not isinstance(path, str):
        raise TypeError("path is not string type")
    if not os.path.exists(path):
        raise FileNotFoundError("File not found with provided path.")

    flag = cv2.IMREAD_COLOR if input_shape[2] == 3 else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)

    if preserve_ratio:
        scale_height = input_shape[0] / img.shape[0]
        scale_width = input_shape[1] / img.shape[1]
        scale_percent = (
            scale_height if scale_height < scale_width else scale_width
        )
        width = int(img.shape[1] * scale_percent)
        height = int(img.shape[0] * scale_percent)
        dim = (width, height)

        # resize image
        resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

        border_height = input_shape[0] - resized.shape[0]
        border_top = int((input_shape[0] - resized.shape[0]) / 2)
        border_bottom = border_height - border_top
        border_width = input_shape[1] - resized.shape[1]
        border_left = int((input_shape[1] - resized.shape[1]) / 2)
        border_right = border_width - border_left

        final_img = cv2.copyMakeBorder(
            resized,
            border_top,
            border_bottom,
            border_left,
            border_right,
            cv2.BORDER_CONSTANT,
        )
    else:
        dim = (input_shape[1], input_shape[0])
        final_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return np.reshape(final_img, input_shape)
