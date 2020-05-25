import os

from nnga.architectures.mlp import MLP
from nnga.architectures.cnn import CNN
from nnga.architectures.cnn_mlp import CNN_MLP

from nnga.datasets.classification.image_dataset import ImageDataset
from nnga.datasets.classification.csv_dataset import CSVDataset
from nnga.datasets.classification.image_csv_dataset import ImageCSVDataset


__version__ = "0.0.0"

ARCHITECTURES = {
    "MLP": MLP,
    "CNN": CNN,
    "CNN/MLP": CNN_MLP,
}

DATASETS = {
    "MLP": CSVDataset,
    "CNN": ImageDataset,
    "CNN/MLP": ImageCSVDataset,
}

ROOT = os.path.abspath(
    os.path.join(os.path.join(__file__, os.pardir), os.pardir)
)
