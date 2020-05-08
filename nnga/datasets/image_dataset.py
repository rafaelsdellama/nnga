import os
from pathlib import Path
from nnga.datasets.base_dataset import BaseDataset
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
            self._dataset_path = os.path.expandvars(cfg.DATASET.VAL_IMG_PATH)
        else:
            self._dataset_path = os.path.expandvars(cfg.DATASET.TRAIN_IMG_PATH)

        super().__init__(cfg, logger, is_validation)

        self.image_shape = cfg.MODEL.INPUT_SHAPE
        self._preserve_ratio = cfg.DATASET.PRESERVE_IMG_RATIO

        if not isinstance(self._preserve_ratio, bool):
            raise ValueError("DATASET.PRESERVE_IMG_RATIO must be a bool")

        if (
            not isinstance(self.image_shape, tuple)
            or len(self.image_shape) != 3
        ):
            raise ValueError(
                "MODEL.INPUT_SHAPE must be a tuple" "(width, height, channels)"
            )

    def _load_metadata(self):
        """Create metadata for classification.
            Metadata is type of index from all files that
            represents dataset. Any huge data is load here
            just create a kind of index.
        """
        data = read_dataset(self._dataset_path, format="png, jpg, jpeg, tif",)

        self._metadata = {
            Path(img_path).name.split(".")[0]: {
                "image_path": img_path,
                "label": label,
            }
            for img_path, label in zip(data["data"], data["labels"])
        }
