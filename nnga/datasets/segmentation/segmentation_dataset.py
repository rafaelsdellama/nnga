import os
import numpy as np
from pathlib import Path
from nnga.utils.data_io import (
    load_image,
    save_encoder_parameters,
    save_decoder_parameters,
)
from nnga.datasets.base_dataset import BaseDataset
from nnga.utils.data_collector import read_dataset
from nnga.utils.data_manipulation import adjust_image_shape, normalize_image


class SegmentationDataset(BaseDataset):
    """SegmentationDataset loader
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
            self._dataset_img_path = os.path.expandvars(
                cfg.DATASET.VAL_IMG_PATH
            )
        else:
            self._dataset_img_path = os.path.expandvars(
                cfg.DATASET.TRAIN_IMG_PATH
            )

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

        super().__init__(cfg, logger, is_validation)

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
        data = read_dataset(
            os.path.join(self._dataset_img_path, 'image'),
            format="png, jpg, jpeg, tif",
        )

        self._metadata = {
            os.path.splitext(Path(img_path).name)[0]: {
                "image_path": img_path,
            }
            for img_path in data["data"]
        }

        mask_data = read_dataset(
            os.path.join(self._dataset_img_path, 'mask'),
            format="png, jpg, jpeg, tif",
        )
        mask_metadata = {
            os.path.splitext(Path(mask_path).name)[0]: {
                "mask_path": mask_path,
            }
            for mask_path in mask_data["data"]
        }

        if set(self._metadata.keys()) != set(mask_metadata.keys()):
            msg = "The files name from image path and mask path does not " \
                  "match!\nCheck the dataset!"
            self._logger.error(msg)
            raise RuntimeError(msg)

        for key in self._metadata.keys():
            self._metadata[key].update(mask_metadata[key])

    def _make_index(self):
        """Using the metadata structure populate attributes
        _indexes, _indexes
        """
        for key, _ in self._metadata.items():
            self._indexes.append(key)

    def _compute_class_weigths(self):
        """Using the metadata structure populate class_weigths
        """
        self._class_weigths = None

    def on_epoch_end(self):
        """
        Update dataset on epoch end
        Notes:
            This method must be called by keras according to
            the keras documentation, but it's not the case in
            tensorflow 2.1.0 implementation, so was needed to
            call this manually
        """
        if self._shuffle:
            np.random.shuffle(self._indexes)

    def _make_labels_info(self):
        self._class_to_id = {"0": 0, "255": 1}
        self._id_to_class = {v: k for k, v in self._class_to_id.items()}

    @property
    def n_classes(self):
        """
            Property number of classes on the dataset

        Returns:
            {int} -- number of classes on the dataset
        """
        return 2

    def save_parameters(self):
        """
        Save parameters from dataset
        """
        save_encoder_parameters(self._output_dir, self._class_to_id)
        save_decoder_parameters(self._output_dir, self._id_to_class)

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
            normalize_image(
                adjust_image_shape(
                    load_image(self._metadata[idx]["image_path"],),
                    self.image_shape,
                )
            )
            for idx in indexes
        ]

        masks = [
            normalize_image(
                adjust_image_shape(
                    load_image(self._metadata[idx]["mask_path"], ),
                    (self.image_shape[0], self.image_shape[1], 1),
                )
            ).reshape(self.image_shape[0],
                      self.image_shape[1])
            for idx in indexes
        ]

        return np.array(images), np.array(masks)
