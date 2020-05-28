import os
import numpy as np
from pathlib import Path
from statistics import mean, stdev
from nnga.datasets.classification.csv_dataset import CSVDataset
from nnga.datasets.classification.image_dataset import ImageDataset
from nnga.utils.data_collector import read_dataset
from nnga.utils.data_io import (
    load_image,
    load_csv_file,
    save_encoder_parameters,
    save_decoder_parameters,
    save_scale_parameters,
)
from nnga.utils.data_manipulation import adjust_image_shape, normalize_image


class ImageCSVDataset(CSVDataset, ImageDataset):
    """ImageCSVDataset loader
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
        super().__init__(cfg, logger, is_validation)

    @property
    def input_shape(self):
        """
            Property is the input shape

        Returns:
            {int or tuple} -- input shape
        """
        return len(self._features), self.image_shape

    def _load_metadata(self):
        """Create metadata for classification.
            Metadata is type of index from all files that
            represents dataset. Any huge data is load here
            just create a kind of index.
        """
        f = open(self._dataset_csv_path)
        self._features = f.readline().replace("\n", "")
        f.close()

        if ";" in self._features:
            self._sep = ";"

        self._features = self._features.split(self._sep)

        col_list = []
        if "class" in self._features:
            col_list.append("class")
            self._features.remove("class")

        if "id" in self._features:
            col_list.append("id")
            self._features.remove("id")
        else:
            raise RuntimeError(
                "The id column does not found!\n" "Check the dataset!"
            )

        data = read_dataset(
            self._dataset_img_path, format="png, jpg, jpeg, tif",
        )
        self._metadata = {
            os.path.splitext(Path(img_path).name)[0]: {
                "image_path": img_path,
                "label": str(label),
            }
            for img_path, label in zip(data["data"], data["labels"])
        }

        _metadata_csv = {}
        for df in load_csv_file(
            self._dataset_csv_path, usecols=col_list, chunksize=10
        ):
            for index, row in df.iterrows():
                _metadata_csv[str(row["id"])] = {
                    "line_path": index,
                }

        if set(self._metadata.keys()) != set(_metadata_csv.keys()):
            if set(self._metadata.keys()).issubset(_metadata_csv.keys()):
                self._logger.warning(
                    f"Some ids from csv is not found on images directory! \n"
                    f"{list(set(_metadata_csv.keys()).difference(self._metadata.keys()))}")
            else:
                msg = "The id column does not match the name of the directory images!\n" \
                      f"Img not in CSV File: " \
                      f"{list(set(self._metadata.keys()).difference(_metadata_csv.keys()))}" \
                      f"\nCheck the dataset!"
                self._logger.error(msg)
                raise RuntimeError(msg)

        for key in self._metadata.keys():
            self._metadata[key].update(_metadata_csv[key])

    def _make_scale_parameters(self):
        """Calculate the scale parameters to be used
        """
        for key in self._features:
            df = load_csv_file(self._dataset_csv_path, usecols=[key])
            df[key] = df[key].astype(float)

            self.scale_parameters[key] = {
                "min": min(df[key].values),
                "max": max(df[key].values),
                "mean": mean(df[key].values),
                "stdev": stdev(df[key].values),
            }

    def save_parameters(self):
        """
        Save parameters from dataset
        """
        save_scale_parameters(self._output_dir, self.scale_parameters)
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
                    self.preserve_ratio,
                )
            )
            for idx in indexes
        ]

        attributes = [self.load_sample_by_idx(idx) for idx in indexes]

        labels = []
        for i, idx in enumerate(indexes):
            np_label = np.zeros(len(self._labels))
            np_label[self.label_encode(self._metadata[idx]["label"])] = 1
            labels.append(np_label)

            if self.features_selected is not None:
                attributes[i] = [
                    attributes[i][index] for index in self.features_selected
                ]

        self._generator_classes.extend(labels)

        return [np.array(attributes), np.array(images)], np.array(labels)
