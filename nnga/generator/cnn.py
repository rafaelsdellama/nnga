import numpy as np
from nnga.generator.base_generator import BaseGenerator
from nnga.utils.data_io import load_image


class CNN_Generator(BaseGenerator):

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
        images = [load_image(
            self._dataset['IMG'].get_metadata_by_idx(idx)['image_path'],
            self._dataset['IMG'].image_shape,
            self._preserve_img_ratio) / 255.0 for idx in indexes]

        labels = []
        for idx in indexes:
            np_label = np.zeros(self._dataset['IMG'].n_classes)
            np_label[self._dataset['IMG'].label_encode(
                self._dataset['IMG'].get_metadata_by_idx(idx)["label"])] = 1
            labels.append(np_label)

        self._classes.extend(labels)

        return np.array(images), np.array(labels)
