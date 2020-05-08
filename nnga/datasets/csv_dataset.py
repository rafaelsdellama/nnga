import os
from statistics import mean, stdev
from nnga.datasets.base_dataset import BaseDataset
from nnga.utils.data_io import load_csv_file
from nnga.utils.scale import scale_features


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
            self._dataset_path = os.path.expandvars(cfg.DATASET.VAL_CSV_PATH)
        else:
            self._dataset_path = os.path.expandvars(cfg.DATASET.TRAIN_CSV_PATH)

        self._data = None
        self._features = None
        self._scale_parameters = {}
        self._sep = ","

        super().__init__(cfg, logger, is_validation)

        self._make_scale_parameters()

    def _load_metadata(self):
        """Create metadata for classification.
            Metadata is type of index from all files that
            represents dataset. Any huge data is load here
            just create a kind of index.
        """
        f = open(self._dataset_path)
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
            self._dataset_path, usecols=col_list, chunksize=10
        ):
            for index, row in df.iterrows():
                self._metadata[row["id"] if "id" in row else index] = {
                    "line_path": index,
                    "label": row["class"] if "class" in row else None,
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
    def n_features(self):
        """
            Property is the number of features in dataset

        Returns:
            {int} -- number of features in dataset
        """
        return len(self._features)

    def _make_scale_parameters(self):
        """Using the metadata structure populate attributes
        _labels, _class_to_id, _id_to_class
        """
        for key in self._features:
            df = load_csv_file(self._dataset_path, usecols=[key])
            df[key] = df[key].astype(float)

            self._scale_parameters[key] = {
                "min": min(df[key].values),
                "max": max(df[key].values),
                "mean": mean(df[key].values),
                "stdev": stdev(df[key].values),
            }

    @property
    def scale_parameters(self):
        """
            Property is the scale parameters for all features in dataset

        Returns:
            {dict} -- scale parameters for all features in dataset
        """
        return self._scale_parameters

    def load_sample_by_idx(self, idx, scale_method):
        """
        Parameters
        ----------
            idx : str
                sample idx to be load
            scale_method: str
                method to scale the data
        Returns
        -------
            Sample data scaled

        """
        f = open(self._dataset_path)
        header = f.readline().replace("\n", "").split(self._sep)
        indexes_of_features = [
            i for i, value in enumerate(header) if value in self._features
        ]
        for _ in range(self._metadata[idx]["line_path"]):
            f.readline()

        sample = f.readline().replace("\n", "").split(self._sep)

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
            sample, header, self._scale_parameters, scale_method=scale_method
        )
        return sample
