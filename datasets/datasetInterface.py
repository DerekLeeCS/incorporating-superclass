import abc
from typing import Dict


class Dataset(abc.ABC):
    """This class defines an interface for retrieving data from a specific dataset."""

    @abc.abstractmethod
    def getData(self, isTraining: bool) -> Dict:
        pass

    @abc.abstractmethod
    def getNumClasses(self) -> int:
        pass
