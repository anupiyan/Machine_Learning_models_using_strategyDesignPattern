import numpy as np
import pandas as pd


from strategy_forML import Strategy_forML
from sklearn.tree import DecisionTreeClassifier


class DecisionTree(Strategy_forML):

    def MLtrainingClassification(self, xtrain:pd.DataFrame, ytrain:pd.DataFrame)->DecisionTreeClassifier:
        dtc = DecisionTreeClassifier()
        model_dtc = dtc.fit(xtrain,ytrain)
        return model_dtc

    def MLtestingClassification(self, model:DecisionTreeClassifier, xtest:pd.DataFrame)->np.ndarray:
        return model.predict(xtest)
