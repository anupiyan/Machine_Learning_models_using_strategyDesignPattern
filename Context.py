import pandas as pd


from strategy_forML import Strategy_forML


class Context():

    def __init__(self, strategy_forML:Strategy_forML)->None:
        self._strategy_forML = strategy_forML

    @property
    def strategy(self)->Strategy_forML:
        return self._strategy_forML

    @strategy.setter
    def strategy(self, strategy_forML: Strategy_forML)->None:
        self._strategy_forML = strategy_forML

    def MLtrainingClassification(self, datax:pd.DataFrame, datay:pd.DataFrame)->Strategy_forML:
        model = self._strategy_forML.MLtrainingClassification(datax, datay)
        return model

    def MLtestingClassification(self, model, xtest:pd.DataFrame):
        result = self._strategy_forML.MLtestingClassification(model, xtest)
        return result
