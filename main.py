import configparser


from ReadCSV import ReadCSV
from sklearn.model_selection import train_test_split
from Decision_tree import DecisionTree
from random_forest import Random_forest
from explainable_ML import explaining_ML
from Context import Context
from Evaluation import Evaluation
from DataPipeline import DataPipeline

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read("config.ini")
    path = config['DATA']['location']
    algorithm = config['MODEL']['algorithm']
    task = config['TASK']['task']
    ReadCSV_obj = ReadCSV(path)
    dataframe = ReadCSV_obj.readcsv()
    if task == "Explainability":
        dataframe['Sex'].replace(['M', 'F'],
                        [0, 1], inplace=True)
        dataframe['ChestPainType'].replace(['ATA', 'NAP','ASY','TA'],
                        [0, 1, 2, 3], inplace=True)
        dataframe['RestingECG'].replace(['Normal', 'ST','LVH'],
                        [0, 1, 2], inplace=True)
        dataframe['ExerciseAngina'].replace(['Y', 'N'],
                        [0, 1], inplace=True)
        dataframe['ST_Slope'].replace(['Down', 'Up', 'Flat'],
                        [0, 1, 2], inplace=True)
        train, test = train_test_split(dataframe, test_size=0.2)
        xtrain = train.iloc[:,:-1]
        xtest = test.iloc[:,:-1]
        ytrain = train.iloc[:,-1:]
        ytest = test.iloc[:,-1:]
        if algorithm =="DecisionTree":
            context = Context(DecisionTree())
        elif algorithm =="RandomForrest":
            context = Context(Random_forest())
        model = context.MLtrainingClassification(xtrain, ytrain)
        results = context.MLtestingClassification(model, xtest)
        exML = explaining_ML(100)
        explainer,shap_values =exML.MLexplainer(model, xtrain)
        exML.plotting_waterfall(shap_values)
        exML.partial_dependency_plotting(model, xtrain, shap_values,"Cholesterol")
        exML.summarized_plot(shap_values)
        exML.model_interpret(xtrain, ytrain, shap_values)

    elif task == "Prediction":
        pre_processing = DataPipeline(dataframe[300:600])
        pre_processing.scalling_data("Amount")
        pre_processing.drop_column("Amount")
        pre_processing.drop_column("Time")
        dataX, dataY = pre_processing.data_split_X_Y()
        xtrain, xtest, ytrain, ytest = pre_processing.training_testing_split(dataX, dataY)
        if algorithm =="DecisionTree":
            context = Context(DecisionTree())
        elif algorithm =="RandomForrest":
            context = Context(Random_forest())
        model = context.MLtrainingClassification(xtrain, ytrain)
        results = context.MLtestingClassification(model, xtest)
        eval = Evaluation(ytest, results)
        eval.calculating_acc()
        eval.classification_result()
        eval.confusion_metrix_result(model, xtest)

    else:
        print("Please check the configuration file")
