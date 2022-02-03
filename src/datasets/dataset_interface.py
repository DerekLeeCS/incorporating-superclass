import abc
from typing import Tuple

import tensorflow as tf


class Dataset(abc.ABC):
    """This class defines an interface for retrieving data from a specific dataset."""

    @abc.abstractmethod
    def get_data(self) -> Tuple[tf.data.TFRecordDataset, tf.data.TFRecordDataset, tf.data.TFRecordDataset]:
        """Return the training, validation, and test datasets, in that order. Each image is in the range of [0, 1]"""
        pass

    @abc.abstractmethod
    def get_batch_size(self) -> int:
        pass

    @abc.abstractmethod
    def get_image_size(self) -> int:
        pass

    @abc.abstractmethod
    def get_num_classes(self) -> Tuple[int, int]:
        """Return the number of subclasses and superclasses, in that order."""
        pass
