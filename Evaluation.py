import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix


class Evaluation:

    def __init__(self, ytest:pd.DataFrame, prediction:np.ndarray)->None:
        self.ytest = ytest
        self.prediction = prediction

    def calculating_acc(self)->None:
        print('Accuracy score is: ',round(accuracy_score(self.ytest, self.prediction),4))

    def classification_result(self)->None:
        print('Classification report: \n',classification_report(self.ytest, self.prediction))
        
    def confusion_metrix_result(self, model, xtest:pd.DataFrame)->None:
        print(type(model))
        print('Confusion Matrix: \n',plot_confusion_matrix(model, xtest, self.ytest))
        plt.show()
