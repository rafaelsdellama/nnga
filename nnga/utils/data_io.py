import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import cv2
import scikitplot as skplt
import csv
from nnga.configs import export_config
import json
# from tensorflow.keras.models import model_from_json
import tensorflow as tf


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


def load_statistic(path):
    """
        Load statistic from GA to continue the execution

        Parameters
        ----------
        path : str
            Directory path

        Returns
        -------
            Pandas DataFrame with the statistic data
    """
    statistic_dir = Path(path, "ga", "results.csv").as_posix()
    if not os.path.exists(statistic_dir):
        raise FileNotFoundError(f"{statistic_dir} does not exist")

    return pd.read_csv(statistic_dir)


def load_pop(path):
    """
        Load pop from GA to continue the execution

        Parameters
        ----------
        path : str
            Directory path

        Returns
        -------
            Pandas DataFrame with the pop data
    """
    pop_dir = Path(path, "ga", "pop.csv").as_posix()
    if not os.path.exists(pop_dir):
        raise FileNotFoundError(f"{pop_dir} does not exist")
    return pd.read_csv(pop_dir)


def save_statistic(path, statistic):
    """
        Save statistic from GA

        Parameters
        ----------
        path : str
            Directory path

        statistic: DataFrame
            List of dict

        Returns
        -------
    """
    out_dir = Path(path, "ga")
    create_dir(out_dir)

    statistic_dir = Path(out_dir, "results.csv").as_posix()
    plot_dir = Path(out_dir, "Generations.png").as_posix()

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


def save_pop(path, pop):
    """
        Save pop from GA

        Parameters
        ----------
        path : str
            Directory path

        pop: DataFrame
            Pandas DataFrame with the pop data

        Returns
        -------
    """
    out_dir = Path(path, "ga")
    create_dir(out_dir)
    pop_dir = Path(out_dir, "pop.csv").as_posix()

    pop_list = []
    fitness_list = []

    for i in range(len(pop.indivs)):
        pop_list.append(pop.indivs[i].chromosome)
        fitness_list.append(pop.indivs[i].fitness)

    last_pop = pd.DataFrame({"indiv": pop_list, "fitness": fitness_list})
    last_pop.to_csv(pop_dir, index=False, header=True)


def save_history(path, model_history):
    """
        Save model history from model

        Parameters
        ----------
        path : str
            Directory path

        model_history: model_history
            model history

        Returns
        -------
    """
    out_dir = Path(path, "metrics")
    create_dir(out_dir)

    legend = []
    plot_dir = Path(out_dir, "history_model.png").as_posix()
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
        np.linspace(
            0,
            len(model_history["loss"]) - 1,
            10
            if len(model_history["loss"]) > 10
            else len(model_history["loss"]),
            dtype=int,
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


def save_cfg(path):
    """
        Save config

        Parameters
        ----------
        path : str
            Directory path

        Returns
        -------
    """
    create_dir(path)
    export_config(Path(path, "config.yaml").as_posix())


def save_roc_curve(path, lbl, predict_proba, labels):
    """
        Save Roc Curve

        Parameters
        ----------
        path : str
            Directory path

        lbl: numoy array
            numpy array with target class

        predict_proba: numoy array
            numpy array with predict proba class

        labels: list
            list with all target label

        Returns
        -------
    """
    out_dir = Path(path, "metrics")
    create_dir(out_dir)

    plot_dir = Path(out_dir, "roc_curve.png").as_posix()
    skplt.metrics.plot_roc([labels[i] for i in lbl], predict_proba)
    plt.savefig(plot_dir)
    plt.clf()
    plt.cla()
    plt.close()


def save_metrics(path, metrics):
    """
        Save all Metrics

        Parameters
        ----------
        path : str
            Directory path

        metrics: Dict
            Dict containing all metrics to be saved

        Returns
        -------
    """
    out_dir = Path(path, "metrics")
    create_dir(out_dir)

    file_name = Path(out_dir, "metrics.txt").as_posix()

    file = open(file_name, "a")

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


def load_csv_line(path, sep, idx):
    """
            Load csv line

            Parameters
            ----------
            path : str
                CSV directory path

            sep: str
                Separator from file

            idx: int
                Row index to be read

            Returns
            -------
            header: list, sample: list

        """
    f = open(path)
    header = f.readline().replace("\n", "").split(sep)

    for _ in range(idx):
        f.readline()

    sample = f.readline().replace("\n", "").split(sep)

    return header, sample


def load_image(path):
    """
        Load image

        Parameters
        ----------
        path : str
            Directory path

        Returns
        -------
        numpy array image

    """

    if not isinstance(path, str):
        raise TypeError("path is not string type")
    if not os.path.exists(path):
        raise FileNotFoundError("File not found with provided path.")

    return cv2.imread(path, cv2.IMREAD_COLOR)


def save_scale_parameters(path, scale_parameters):
    """
        Save scale parameters

        Parameters
        ----------
        path : str
            Directory path

        scale_parameters: Dict
            Dict containing the scale parameters

        Returns
        -------
    """
    out_dir = Path(path, "dataset")
    create_dir(out_dir)
    file_name = Path(out_dir, "scale_parameters.json").as_posix()

    file = open(file_name, "w")
    json.dump(scale_parameters, file)
    file.close()


def save_feature_selected(path, feature_selected):
    """
        Save features selected

        Parameters
        ----------
        path : str
            Directory path

        feature_selected: List
            List with features selected

        Returns
        -------
    """
    out_dir = Path(path, "dataset")
    create_dir(out_dir)
    file_name = Path(out_dir, "feature_selected.json").as_posix()

    file = open(file_name, "w")
    json.dump(feature_selected, file)
    file.close()


def save_encoder_parameters(path, encode):
    """
    Save Encode parameters

    Parameters
    ----------
        path : str
            Directory path

        encode: Dict
            Encode parameters


    Returns
    -------
    """
    out_dir = Path(path, "dataset")
    create_dir(out_dir)

    file_name = Path(out_dir, "encode.json").as_posix()
    file = open(file_name, "w")
    json.dump(encode, file)
    file.close()


def save_decoder_parameters(path, decode):
    """
    Save Decode parameters

    Parameters
    ----------
        path : str
            Directory path

        decode: Dict
            Decode parameters

    Returns
    -------
    """
    out_dir = Path(path, "dataset")
    create_dir(out_dir)

    file_name = Path(out_dir, "decode.json").as_posix()
    file = open(file_name, "w")
    json.dump(decode, file)
    file.close()


def save_model(path, model):
    """
        Save model

        Parameters
        ----------
        path : str
            Directory path

        model: model
            model to be saved

        Returns
        -------
    """
    create_dir(path)
    model.save(path)


def load_scale_parameters(path):
    """
        Load scale parameters

        Parameters
        ----------
        path : str
            Directory path

        Returns
        -------
            scale_parameters: Dict
                Dict containing the scale parameters
        """
    out_dir = Path(path, "dataset")
    file_name = Path(out_dir, "scale_parameters.json").as_posix()

    with open(file_name) as json_file:
        scale_parameters = json.load(json_file)

    return scale_parameters


def load_feature_selected(path):
    """
        Load features selected

        Parameters
        ----------
        path : str
            Directory path

        Returns
        -------
        feature_selected: List
            List with features selected
    """
    out_dir = Path(path, "dataset")
    file_name = Path(out_dir, "feature_selected.json").as_posix()

    with open(file_name) as json_file:
        feature_selected = json.load(json_file)

    return feature_selected


def load_encoder_parameters(path):
    """
    Load Encoder parameters

    Parameters
    ----------
        path : str
            Directory path

    Returns
    -------
        encode: Dict
            Encode parameters

    """
    out_dir = Path(path, "dataset")

    file_name = Path(out_dir, "encode.json").as_posix()
    with open(file_name) as json_file:
        encode = json.load(json_file)

    return encode


def load_decoder_parameters(path):
    """
    Load Decoder parameters

    Parameters
    ----------
        path : str
            Directory path

    Returns
    -------

        decode: Dict
            Decode parameters
    """
    out_dir = Path(path, "dataset")

    file_name = Path(out_dir, "decode.json").as_posix()
    with open(file_name) as json_file:
        decode = json.load(json_file)

    return decode


def load_model(path):
    """
        Load model

        Parameters
        ----------
        path : str
            Path to directory with the saved model

        Returns
        -------
        model: model
            model read
    """
    return tf.keras.models.load_model(path)


def save_train_state(path, train_state):
    """
    Save Train state

    Parameters
    ----------
        path : str
            Directory path

        train_state: Dict
            train_state parameters

    Returns
    -------
    """
    create_dir(path)

    file_name = Path(path, "train_state.json").as_posix()
    file = open(file_name, "w")
    json.dump(train_state, file)
    file.close()


def load_train_state(path):
    """
    Load train_state parameters

    Parameters
    ----------
        path : str
            Directory path

    Returns
    -------
        train_state: Dict
            train_state parameters

    """

    file_name = Path(path, "train_state.json").as_posix()
    with open(file_name) as json_file:
        train_state = json.load(json_file)

    return train_state
