import os
import numpy as np
from pathlib import Path
from nnga.utils.data_io import load_image
from nnga.datasets.classification.base_dataset import BaseDataset
from nnga.utils.data_collector import read_dataset


class ImageDataset(BaseDataset):
    """ImageDataset loader
          class implements {BaseDataset} and provide a loader
    Arguments
        cfg : {yacs.config.CfgNode}
            Experiment config data.
            All the information to configure the dataset loader is stored
            in experiment config.
        is_validation : {bool}
            Flag to sinalize when dataset loader is a validation dataset
            it's important select information on experiment
            config to configure dataset loader corretly for validation
      """

    def __init__(self, cfg, logger, is_validation=False):
        self._is_validation = is_validation
        if self._is_validation:
            self._dataset_img_path = os.path.expandvars(cfg.DATASET.VAL_IMG_PATH)
        else:
            self._dataset_img_path = os.path.expandvars(cfg.DATASET.TRAIN_IMG_PATH)

        super().__init__(cfg, logger, is_validation)

        self.image_shape = cfg.MODEL.INPUT_SHAPE
        self.preserve_ratio = cfg.DATASET.PRESERVE_IMG_RATIO

        if not isinstance(self.preserve_ratio, bool):
            raise ValueError("DATASET.PRESERVE_IMG_RATIO must be a bool")

        if (
            not isinstance(self.image_shape, tuple)
            or len(self.image_shape) != 3
        ):
            raise ValueError(
                "MODEL.INPUT_SHAPE must be a tuple" "(width, height, channels)"
            )

    @property
    def input_shape(self):
        """
            Property is the input shape

        Returns:
            {int or tuple} -- input shape
        """
        return self.image_shape

    def _load_metadata(self):
        """Create metadata for classification.
            Metadata is type of index from all files that
            represents dataset. Any huge data is load here
            just create a kind of index.
        """
        data = read_dataset(self._dataset_img_path, format="png, jpg, jpeg, tif",)

        self._metadata = {
            Path(img_path).name.split(".")[0]: {
                "image_path": img_path,
                "label": label,
            }
            for img_path, label in zip(data["data"], data["labels"])
        }

    def _data_generation(self, indexes):
        """
        Method use indexes to generate a batch data

        Use metadata structure to:
                - load images and ground thruth
                - apply data augmentation
                - form batchs

        Parameters
        ----------
            indexes {list} -- list of indexes from metadata to be
                loaded in a bacth with input and ground thruth
        """
        # TODO: Data augmentation
        images = [
            load_image(
                self._metadata[idx]["image_path"],
                self.image_shape,
                self.preserve_ratio,
            )
            / 255.0
            for idx in indexes
        ]

        labels = []
        for idx in indexes:
            np_label = np.zeros(len(self._labels))
            np_label[
                self.label_encode(
                    self._metadata[idx]["label"]
                )
            ] = 1
            labels.append(np_label)

        self._generator_classes.extend(labels)

        return np.array(images), np.array(labels)
