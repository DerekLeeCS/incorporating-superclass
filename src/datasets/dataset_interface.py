import abc
from typing import Dict


class Dataset(abc.ABC):
    """This class defines an interface for retrieving data from a specific dataset."""

    @abc.abstractmethod
    def get_data(self, is_training: bool) -> Dict:
        """Returns a dictionary containing the images and labels. The images are normalized from 0 to 1."""
        pass

    @abc.abstractmethod
    def get_num_classes(self) -> int:
        pass