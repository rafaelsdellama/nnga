import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
import cv2


def load_statistic(path, seed):
    statistic_dir = Path(path, f"resuts_exec_{seed}.csv").as_posix()
    if not os.path.exists(statistic_dir):
        raise FileNotFoundError(f"{statistic_dir} does not exist")

    return pd.read_csv(statistic_dir)


def load_pop(path, seed):
    pop_dir = Path(path, f"pop_exec_{seed}.csv").as_posix()
    if not os.path.exists(pop_dir):
        raise FileNotFoundError(f"{pop_dir} does not exist")
    return pd.read_csv(pop_dir)


def save_statistic(path, seed, statistic):
    statistic_dir = Path(path, f"resuts_exec_{seed}.csv").as_posix()
    plot_dir = Path(path, f"plot_exec_{seed}.png").as_posix()

    df = pd.DataFrame(
        statistic,
        columns=['mean_fitness', 'best_fitness', 'best_indiv']
    )
    df.to_csv(statistic_dir, index=False, header=True)

    plt.plot(list(df['mean_fitness']), 'r')
    plt.plot(list(df['best_fitness']), 'g')
    plt.xlabel("Generations")
    plt.ylabel("Fitness")
    plt.title("Results exec {}".format(exec))
    plt.legend(['Mean Fitness', 'Best Fitness'])

    plt.savefig(plot_dir)
    plt.clf()  # Clear the current figure.
    plt.cla()  # Clear the current axes.
    plt.close()


def save_pop(path, seed, pop):
    pop_list = []
    fitness_list = []
    pop_dir = Path(path, f"pop_exec_{seed}.csv").as_posix()

    for i in range(len(pop.indivs)):
        pop_list.append(pop.indivs[i].chromosome)
        fitness_list.append(pop.indivs[i].fitness)

    last_pop = pd.DataFrame({'indiv': pop_list, 'fitness': fitness_list})
    last_pop.to_csv(pop_dir, index=False, header=True)


def load_csv_file(path, usecols, sep=';', decimal=","):
    """
    :param path: csv path
    :param sep: separator of csv
    :return: Dataframe containing: labels | data(s) | classes
    """
    if not isinstance(path, str):
        raise TypeError("path is not string type")
    if not os.path.exists(path):
        raise FileNotFoundError("File not found with provided path.")

    return pd.read_csv(path, usecols=usecols, sep=sep, decimal=decimal)


def load_image(path, input_shape, preserve_ratio=True):

    if not isinstance(path, str):
        raise TypeError("path is not string type")
    if not os.path.exists(path):
        raise FileNotFoundError("File not found with provided path.")

    flag = cv2.IMREAD_COLOR \
        if input_shape[2] == 3 \
        else cv2.IMREAD_GRAYSCALE
    img = cv2.imread(path, flag)

    if preserve_ratio:
        scale_height = input_shape[0] / img.shape[0]
        scale_width = input_shape[1] / img.shape[1]
        scale_percent = scale_height \
            if scale_height < scale_width else scale_width
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

        final_img = cv2.copyMakeBorder(resized,
                                       border_top,
                                       border_bottom,
                                       border_left,
                                       border_right,
                                       cv2.BORDER_CONSTANT)
    else:
        dim = (input_shape[1], input_shape[0])
        final_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return np.reshape(final_img, input_shape)
