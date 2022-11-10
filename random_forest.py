import numpy as np
import pandas as pd


from strategy_forML import Strategy_forML
from sklearn.ensemble import RandomForestClassifier


class Random_forest(Strategy_forML):
    def MLtrainingClassification(self, xtrain:pd.DataFrame, ytrain:pd.DataFrame)->RandomForestClassifier:
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        model_clf = clf.fit(xtrain, ytrain.values.ravel())
        return model_clf

    def MLtestingClassification(self, model:RandomForestClassifier, xtest:pd.DataFrame)->np.ndarray:
        return model.predict(xtest)

