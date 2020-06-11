import os

from nnga.architectures.classification.mlp import MLP
from nnga.architectures.classification.cnn import CNN
from nnga.architectures.classification.cnn_mlp import CNN_MLP
from nnga.architectures.segmentation.segmentation import Segmentation

from nnga.datasets.classification.image_dataset import ImageDataset
from nnga.datasets.classification.csv_dataset import CSVDataset
from nnga.datasets.classification.image_csv_dataset import ImageCSVDataset
from nnga.datasets.segmentation.segmentation_dataset import SegmentationDataset


__version__ = "1.1.0"

ARCHITECTURES = {
    "Classification": {
        "MLP": MLP,
        "CNN": CNN,
        "CNN/MLP": CNN_MLP,
    },
    "Segmentation": {
        "CNN": Segmentation,
    }
}

DATASETS = {
    "Classification": {
        "MLP": CSVDataset,
        "CNN": ImageDataset,
        "CNN/MLP": ImageCSVDataset,
    },
    "Segmentation": {
        "CNN": SegmentationDataset,
    }
}

ROOT = os.path.abspath(
    os.path.join(os.path.join(__file__, os.pardir), os.pardir)
)


def get_dataset(task, architecture):
    """Return a dataset loader for a specific task and architecture"""
    dataset_by_task = DATASETS.get(task)
    dataset = dataset_by_task.get(architecture)
    if dataset_by_task is None:
        raise RuntimeError(
            f"There isn't a valid TASKS!\n Check your experiment config\n"
            f"Tasks: {DATASETS.keys()}"
        )

    elif dataset is None:
        raise RuntimeError(
            f"There isn't a valid architecture configured!\nCheck your "
            f"experiment config\nArchitecture for {task}: "
            f"{DATASETS.get(task).keys()}"
        )

    return dataset


def get_architecture(task, architecture):
    """Return a architecture for a specific task and architecture"""
    architecture_by_task = ARCHITECTURES.get(task)
    architecture = architecture_by_task.get(architecture)
    if architecture_by_task is None:
        raise RuntimeError(
            f"There isn't a valid TASKS!\n Check your experiment config\n"
            f"Tasks: {ARCHITECTURES.keys()}"
        )

    elif architecture is None:
        raise RuntimeError(
            f"There isn't a valid architecture configured!\nCheck your "
            f"experiment config\nArchitecture for {task}: "
            f"{ARCHITECTURES.get(task).keys()}"
        )

    return architecture
