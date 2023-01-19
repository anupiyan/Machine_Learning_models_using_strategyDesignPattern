from ReadCSV import ReadCSV
from sklearn.model_selection import train_test_split
from Decision_tree import DecisionTree
from random_forest import Random_forest

from explainable_ML import explaining_ML
from Context import Context
from sklearn.tree import DecisionTreeClassifier

path = './heart.csv'
ReadCSV_obj = ReadCSV(path)
dataframe = ReadCSV_obj.readcsv()

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

context = Context(DecisionTree())
model = context.MLtrainingClassification(xtrain, ytrain)
results = context.MLtestingClassification(model, xtest)

exML = explaining_ML(100)
explainer,shap_values =exML.MLexplainer(model, xtrain)
exML.plotting_waterfall(shap_values)
exML.partial_dependency_plotting(model, xtrain, shap_values,"Cholesterol")
exML.summarized_plot(shap_values)
exML.model_interpret(xtrain, ytrain, shap_values)
