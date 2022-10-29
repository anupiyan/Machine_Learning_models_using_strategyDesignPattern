
from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix
import matplotlib.pyplot as plt

class Evaluation:

    def __init__(self, ytest, prediction):
        self.ytest = ytest
        self.prediction = prediction

    def calculating_acc(self):
        print('Accuracy score is: ',round(accuracy_score(self.ytest, self.prediction),4))

    def classification_result(self):
        print('Classification report: \n',classification_report(self.ytest, self.prediction))
        
    def confusion_metrix_result(self, model, xtest):
        print('Confusion Matrix: \n',plot_confusion_matrix(model, xtest, self.ytest))
        plt.show()
