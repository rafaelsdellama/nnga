import random
import numpy as np
from tensorflow.keras.utils import Sequence


class BaseDataset(Sequence):
    """This Class implements tensorflow.keras.utils.Sequence to be compatible
        with keras training loop implementing the following methods
        (on_epoch_end, __len__ and __getitem__)

    Arguments:
        cfg {yacs.config import CfgNode} -- Experiment config data
            All the information to configure the dataset loader is stored
            in experiment config
        is_validation {bool} -- Flag to sinalize when dataset loader is a
            validation dataset it's important select information on experiment
            config to configure dataset loader corretly for validation

    """

    def __init__(self, cfg, logger, is_validation=False):

        self._metadata = {}
        self._indexes = []
        self._indexes_labels = []
        self._generator_classes = []
        self._class_to_id = None
        self._id_to_class = None

        self._labels = []
        self._class_weigths = {}

        self._is_validation = is_validation
        if self._is_validation:
            self._augmentation = cfg.DATASET.VAL_AUGMENTATION
            self._shuffle = cfg.DATASET.VAL_SHUFFLE
        else:
            self._augmentation = cfg.DATASET.TRAIN_AUGMENTATION
            self._shuffle = cfg.DATASET.TRAIN_SHUFFLE

        self._batch_size = cfg.SOLVER.BATCH_SIZE
        self._output_dir = cfg.OUTPUT_DIR

        if not isinstance(self._augmentation, bool):
            raise ValueError("DATASET.___AUGMENTATION must be a bool")

        if not isinstance(self._shuffle, bool):
            raise ValueError("DATASET.___SHUFFLE must be a bool")

        self._load_metadata()
        self._make_labels_info()
        self._make_index()
        self._compute_class_weigths()
        self._make_scale_parameters()

        self.on_epoch_end()

        if not self._is_validation:
            self.save_parameters()

    def _load_metadata(self):
        """Create metadata for classification.
            Metadata is type of index from all files that
            represents dataset. Any huge data is load here
            just create a kind of index.
        """
        pass

    def _make_labels_info(self):
        """Using the metadata structure populate attributes
        _labels, _class_to_id, _id_to_class
        """
        unique_labels = list(
            sorted(
                set([value["label"] for _, value in self._metadata.items()])
            )
        )
        self._labels = unique_labels

        self._class_to_id = {l: i for i, l in enumerate(self._labels)}
        self._id_to_class = {v: k for k, v in self._class_to_id.items()}

    @property
    def n_samples(self):
        """Return number of samples in the dataset

        Returns:
            {int} -- Number of samples in the dataset
        """
        return len(self._indexes)

    def _compute_class_weigths(self):
        """Using the metadata structure populate class_weigths
        """
        class_count = {l: 0 for l in self._labels}

        for _, value in self._metadata.items():
            class_count[value["label"]] += 1

        max_count = max([v for _, v in class_count.items()])
        class_weigths_map = {
            k: (v / max_count) for k, v in class_count.items()
        }

        self._class_weigths = {
            self._class_to_id[l]: class_weigths_map[l] for l in self._labels
        }

    @property
    def labels(self):
        """
            Property is list with all labels in dataset

        Returns:
            {list} -- list with all labels in dataset
        """
        return self._labels

    @property
    def n_classes(self):
        """
            Property number of classes on the dataset

        Returns:
            {int} -- number of classes on the dataset
        """
        return len(self._labels)

    @property
    def class_weights(self):
        """
            Property dict to map class indexes to class weights

        Returns:
             {dict} -- dict to map class indexes to class weights
        """
        return self._class_weigths

    def _make_index(self):
        """Using the metadata structure populate attributes
        _indexes, _indexes
        """
        for key, value in self._metadata.items():
            self._indexes.append(key)
            self._indexes_labels.append(value["label"])

    def set_index(self, index):
        try:
            self._indexes = list(np.array(self._indexes)[index])
            self._indexes_labels = list(np.array(self._indexes_labels)[index])
        except Exception:
            raise ValueError(
                f"Some index is not a "
                f"valid index on dataset! Error: {Exception}"
            )

    @property
    def indexes(self):
        """
            Property is list with all index in dataset

        Returns:
            (list, list) --
            list with all index in dataset
            list with all indexes_labels in dataset
        """
        return self._indexes, self._indexes_labels

    def label_encode(self, label):
        """
        Encode label
        Parameters
        ----------
            label : str
                Class label
        Returns
        -------
            str label encoded

        """
        return self._class_to_id[label]

    def label_decode(self, label):
        """
        Decode the label
        Parameters
        ----------
            label : str
                label encoded
        Returns
        -------
            str label decoded

        """
        return self._id_to_class[label]

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
            c = list(zip(self._indexes, self._indexes_labels))
            np.random.shuffle(c)
            self._indexes[:], self._indexes_labels[:] = zip(*c)

    def __len__(self):
        """Return number of batchs in the dataset

        Returns:
            {int} -- Number of batchs in the dataset
        """
        return int(np.ceil(len(self._indexes) / float(self._batch_size)))

    def __getitem__(self, idx):
        """Get a batch from dataset at index

        Arguments:
            index {int} -- Position to get batch

        Returns:
             {tuple} -- Batch data with inputs and ground thruth
        """
        # Generate indexes of the batch
        indexes = list(
            self._indexes[
                idx * self._batch_size : (idx + 1) * self._batch_size
            ]
        )

        data = self._data_generation(indexes)

        if (idx + 1) == self.__len__():
            self.on_epoch_end()

        return data

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
        pass

    @property
    def input_shape(self):
        """
            Property is the input shape

        Returns:
            {int or tuple} -- input shape
        """
        pass

    @property
    def classes(self):
        """
            Property is the classes list generated by the Generator

        Returns:
            {list} -- classes list generated by the Generator
        """
        return self._generator_classes

    @property
    def num_indexes(self):
        """
            Property is the number of index in dataset used by the generator

        Returns:
            {int} -- number of index in dataset used by the generator
        """
        return len(self._indexes)

    def reset_generator(self):
        """
            Reset the generator
        """
        self.on_epoch_end()
        self._generator_classes = []

    def save_parameters(self):
        """
        Save parameters from dataset
        """
        pass

    def _make_scale_parameters(self):
        """Calculate the scale parameters to be used
        """
