import os
import numpy as np
from statistics import mean, stdev
from nnga.datasets.base_dataset import BaseDataset
from nnga.utils.data_io import (
    load_csv_file,
    load_csv_line,
    save_scale_parameters,
    save_encoder_parameters,
    save_decoder_parameters,
)
from nnga.utils.data_manipulation import scale_features


class CSVDataset(BaseDataset):
    """CSVDataset loader
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
            self._dataset_csv_path = os.path.expandvars(
                cfg.DATASET.VAL_CSV_PATH
            )
        else:
            self._dataset_csv_path = os.path.expandvars(
                cfg.DATASET.TRAIN_CSV_PATH
            )

        self._data = None
        self._features = None
        self.scale_parameters = {}
        self._sep = ","
        self._scale_method = cfg.DATASET.SCALER
        self.features_selected = None

        super().__init__(cfg, logger, is_validation)

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
        else:
            raise RuntimeError(
                "The class column does not found!\n" "Check the dataset!"
            )

        if "id" in self._features:
            col_list.append("id")
            self._features.remove("id")

        for df in load_csv_file(
            self._dataset_csv_path, usecols=col_list, chunksize=10
        ):
            for index, row in df.iterrows():
                self._metadata[index] = {
                    "line_path": index,
                    "label": str(row["class"]),
                }

    @property
    def features(self):
        """
            Property is list with all features in dataset

        Returns:
            {list} -- list with all features in dataset
        """
        return self._features

    @property
    def input_shape(self):
        """
            Property is the input shape

        Returns:
            {int} -- input shape
        """
        return len(self._features)

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

    def load_sample_by_idx(self, idx):
        """
        Parameters
        ----------
            idx : str
                sample idx to be load
        Returns
        -------
            Sample data scaled

        """
        header, sample = load_csv_line(
            self._dataset_csv_path, self._sep, self._metadata[idx]["line_path"]
        )

        indexes_of_features = [
            i for i, value in enumerate(header) if value in self._features
        ]

        sample = [
            float(value.replace(",", "."))
            for i, value in enumerate(sample)
            if i in indexes_of_features
        ]
        header = [
            value for i, value in enumerate(header) if i in indexes_of_features
        ]

        # Normalize the data
        sample = scale_features(
            sample,
            header,
            self.scale_parameters,
            scale_method=self._scale_method,
        )
        return sample

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

        return np.array(attributes), np.array(labels)
