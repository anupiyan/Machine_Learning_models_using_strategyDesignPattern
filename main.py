from ReadCSV import ReadCSV
from DataPipeline import DataPipeline
from Context import Context
from Evaluation import Evaluation
from Decision_tree import DecisionTree
from random_forest import Random_forest


if __name__=="__main__":
    doc = ReadCSV("./creditcard.csv")
    data = doc.readcsv()
    pre_processing = DataPipeline(data)
    pre_processing.scalling_data("Amount")
    pre_processing.drop_column("Amount")
    pre_processing.drop_column("Time")
    dataX, dataY = pre_processing.data_split_X_Y()
    xtrain, xtest, ytrain, ytest = pre_processing.training_testing_split(dataX, dataY)

    #Decision Trees
    print("===Decision Tree===")
    context = Context(DecisionTree())
    model = context.MLtrainingClassification(xtrain, ytrain)
    results = context.MLtestingClassification(model, xtest)
    eval = Evaluation(ytest, results)
    eval.calculating_acc()
    eval.classification_result()
    eval.confusion_metrix_result(model, xtest)

    
    #Random Forest
    print("===Random Forest===")
    context2 = Context(Random_forest())
    model = context2.MLtrainingClassification(xtrain, ytrain)
    results = context2.MLtestingClassification(model, xtest)
    eval = Evaluation(ytest, results)
    eval.calculating_acc()
    eval.classification_result()
    eval.confusion_metrix_result(model, xtest)
