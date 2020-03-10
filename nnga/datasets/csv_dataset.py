import os
from statistics import mean, stdev
from nnga.datasets.base_dataset import BaseDataset
from nnga.utils.data_io import load_csv_file
from nnga.utils.scale import scale_features


class CSVDataset(BaseDataset):

    def __init__(self, cfg, logger, is_validation=False):
        self._is_validation = is_validation
        if self._is_validation:
            self._dataset_path = os.path.expandvars(cfg.DATASET.VAL_CSV_PATH)
        else:
            self._dataset_path = os.path.expandvars(cfg.DATASET.TRAIN_CSV_PATH)

        self._data = None
        self._features = None
        self._scale_parameters = {}

        super().__init__(cfg, logger, is_validation)

        self._make_scale_parameters()

    def _load_metadata(self):
        """Create metadata for classification.
            Metadata is type of index from all files that
            represents dataset. Any huge data is load here
            just create a kind of index.
        """
        f = open(self._dataset_path)
        self._features = f.readline().replace("\n", "").split(';')

        col_list = ["class"]
        self._features.remove('class')

        if 'id' in self._features:
            self._features.remove('id')
            col_list.append('id')

        df = load_csv_file(self._dataset_path, usecols=col_list)

        self._metadata = {
            row['id'] if 'id' in row else index:
                {"line_path": index, "label": row['class']}
            for index, row in df.iterrows()
        }

    @property
    def features(self):
        return self._features

    @property
    def n_features(self):
        return len(self._features)

    def _make_scale_parameters(self):
        for key in self._features:
            df = load_csv_file(self._dataset_path, usecols=[key])

            self._scale_parameters[key] = {
                'min': min(df[key]),
                'max': max(df[key]),
                'mean': mean(df[key]),
                'stdev': stdev(df[key])
            }

    @property
    def scale_parameters(self):
        return self._scale_parameters

    def load_sample_by_idx(self, idx, scale_method):
        f = open(self._dataset_path)
        header = f.readline().replace("\n", "").split(';')
        indexes_of_features = [i for i, value in enumerate(header)
                               if value in self._features]
        for i in range(self._metadata[idx]['line_path']):
            f.readline()

        sample = f.readline().replace("\n", "").split(';')

        sample = [value for i, value in enumerate(sample)
                  if i in indexes_of_features]
        header = [value for i, value in enumerate(header)
                  if i in indexes_of_features]

        # Normalize the data
        sample = scale_features(sample, header,
                                self._scale_parameters,
                                scale_method=scale_method)
        return sample
