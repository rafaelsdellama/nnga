from tensorflow.keras.utils import Sequence
import numpy as np
import random


class BaseGenerator(Sequence):

    def __init__(self, dataset, indexes, batch_size, shuffle, attributes,
                 scale_method, preserve_img_ratio):
        self._dataset = dataset
        self._batch_size = batch_size
        self._indexes = indexes
        self._shuffle = shuffle
        self._attributes_selected = attributes
        self._scale_method = scale_method
        self._classes = []
        self._preserve_img_ratio = preserve_img_ratio

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
            self._indexes[idx * self._batch_size:
                          (idx + 1) * self._batch_size])

        # to handle with the end of epoch complete the rest of dataset
        while len(indexes) < self._batch_size:
            indexes.append(indexes[random.randrange(len(indexes))])
        data = self._data_generation(indexes)

        if (idx + 1) == self.__len__():
            self.on_epoch_end()

        return data

    def _data_generation(self, indexes):
        """
        Method use indexes to generate a batch data

        Arguments:
            indexes {list} -- list of indexes from metadata to be
                loaded in a bacth with input and ground thruth
        """
        pass

    @property
    def classes(self):
        return self._classes

    @property
    def num_indexes(self):
        return len(self._indexes)

    def reset_generator(self):
        self.on_epoch_end()
        self._classes = []
