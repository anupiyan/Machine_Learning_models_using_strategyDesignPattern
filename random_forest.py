from strategy_forML import Strategy_forML
from sklearn.ensemble import RandomForestClassifier


class Random_forest(Strategy_forML):
    def MLtrainingClassification(self, xtrain, ytrain):
        clf = RandomForestClassifier(max_depth=2, random_state=0)
        model_clf = clf.fit(xtrain, ytrain)
        return model_clf

    def MLtestingClassification(self, model, xtest):
        return model.predict(xtest)

