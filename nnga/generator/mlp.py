import numpy as np
from nnga.generator.base_generator import BaseGenerator


class MLP_Generator(BaseGenerator):

    def __init__(self, dataset, indexes, batch_size, shuffle, attributes,
                 scale_method, preserve_img_ratio):
        super().__init__(dataset, indexes, batch_size, shuffle, attributes,
                         scale_method, preserve_img_ratio)

    def _data_generation(self, indexes):
        """
        Method use indexes to generate a batch data

        Arguments:
            indexes {list} -- list of indexes from metadata to be
                loaded in a bacth with input and ground thruth
        """
        attributes = [
            self._dataset['CSV'].load_sample_by_idx(
                idx,
                self._scale_method)
            for idx in indexes
        ]

        labels = []
        for i, idx in enumerate(indexes):
            np_label = np.zeros(self._dataset['CSV'].n_classes)
            np_label[self._dataset['CSV'].label_encode(
                self._dataset['CSV'].get_metadata_by_idx(idx)["label"])] = 1
            labels.append(np_label)

            attributes[i] = [attributes[i][index]
                             for index in self._attributes_selected]

        self._classes.extend(labels)

        return np.array(attributes), np.array(labels)
