import pandas as pd

from abc import ABC, abstractmethod


class Strategy_forML(ABC):
    """
    This is a the strategy abstract class for ML models
    """

    @abstractmethod
    def MLtrainingClassification(self, xtrain:pd.DataFrame, ytrain:pd.DataFrame):
        pass

    @abstractmethod
    def MLtestingClassification(self, model, xtest:pd.DataFrame):
        pass




    
