import os
import numpy as np
from pathlib import Path
from nnga.utils.data_io import load_image, load_csv_file
from nnga.datasets.classification.csv_dataset import CSVDataset
from nnga.datasets.classification.image_dataset import ImageDataset
from nnga.utils.data_collector import read_dataset
from nnga.utils.data_io import load_csv_file, save_feature_selected


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

        for df in load_csv_file(
                self._dataset_csv_path, usecols=col_list, chunksize=10
        ):
            for index, row in df.iterrows():
                self._metadata[row["id"] if "id" in row else index] = {
                    "line_path": index,
                    "label": row["class"] if "class" in row else None,
                }

        data = read_dataset(self._dataset_img_path, format="png, jpg, jpeg, tif",)

        _metadata_aux = {
            Path(img_path).name.split(".")[0]: {
                "image_path": img_path,
                "label": label,
            }
            for img_path, label in zip(data["data"], data["labels"])
        }
        if self._metadata.keys() != _metadata_aux.keys():
            raise RuntimeError(
                "The id column does not match the name of the directory images!\n"
                "Check the dataset!"
            )

        for key in self._metadata.keys():
            self._metadata[key].update(_metadata_aux[key])

    def save_parameters(self, path, seed=''):
        """
        Save parameters from dataset

        Parameters
        ----------
        path : str
            Directory path
        seed: int
            Seed from GA
        """
        if self.attributes_selected is not None:
            save_feature_selected(path, [self._features[i] for i in self.attributes_selected], seed)

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

        attributes = [
            self.load_sample_by_idx(idx)
            for idx in indexes
        ]

        labels = []
        for i, idx in enumerate(indexes):
            np_label = np.zeros(len(self._labels))
            np_label[
                self.label_encode(
                    self._metadata[idx]["label"]
                )
            ] = 1
            labels.append(np_label)

            if self.attributes_selected is not None:
                attributes[i] = [
                    attributes[i][index] for index in self.attributes_selected
                ]

        self._generator_classes.extend(labels)

        return [np.array(attributes), np.array(images)], np.array(labels)
