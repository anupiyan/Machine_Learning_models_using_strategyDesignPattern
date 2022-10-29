from strategy_forML import Strategy_forML
from sklearn.tree import DecisionTreeClassifier
class DecisionTree(Strategy_forML):

    def MLtrainingClassification(self, xtrain, ytrain):
        dtc = DecisionTreeClassifier()
        model_dtc = dtc.fit(xtrain,ytrain)
        return model_dtc

    def MLtestingClassification(self, model, xtest):
        return model.predict(xtest)
