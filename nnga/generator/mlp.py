import numpy as np
from nnga.generator.base_generator import BaseGenerator


class MLP_Generator(BaseGenerator):
    """This Class implements tensorflow.keras.utils.Sequence to be compatible
        with keras training loop implementing the following methods
        (on_epoch_end, __len__ and __getitem__)

    Arguments:
        dataset: BaseDataset
            dataset to be used on the Generator
        indexes: list
            list of index from data
        batch_size: int
            batch size to be used to generate the batch
        shuffle: bool
            if the data will be shuffle after each epoch
        attributes: list
            attributes to be returned
        scale_method: str
            method to scale the data
        preserve_img_ratio: str
            if True, preserve image ratio,
            else Does not preserve the image ratio
    """

    def __init__(
        self,
        dataset,
        indexes,
        batch_size,
        shuffle,
        attributes,
        scale_method,
        preserve_img_ratio,
    ):
        super().__init__(
            dataset,
            indexes,
            batch_size,
            shuffle,
            attributes,
            scale_method,
            preserve_img_ratio,
        )

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

        attributes = [
            self._dataset["CSV"].load_sample_by_idx(idx, self._scale_method)
            for idx in indexes
        ]

        labels = []
        for i, idx in enumerate(indexes):
            np_label = np.zeros(self._dataset["CSV"].n_classes)
            np_label[
                self._dataset["CSV"].label_encode(
                    self._dataset["CSV"].get_metadata_by_idx(idx)["label"]
                )
            ] = 1
            labels.append(np_label)

            if self._attributes_selected is not None:
                attributes[i] = [
                    attributes[i][index] for index in self._attributes_selected
                ]

        self._classes.extend(labels)

        return np.array(attributes), np.array(labels)
