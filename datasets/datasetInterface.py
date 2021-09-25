import abc
from typing import Dict

'''This class defines an interface for retrieving data from a specific dataset'''
class Dataset(abc.ABC):
    @abc.abstractmethod
    def getData(self, isTraining: bool) -> Dict:
        pass