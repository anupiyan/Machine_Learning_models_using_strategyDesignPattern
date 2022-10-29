import pandas as pd
import numpy as np
import seaborn as sb

import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('./creditcard.csv')

corr = data.corr()
ax = sb.heatmap(corr, annot=True, cmap="YlGnBu")

## Display the visualization of the Confusion Matrix.
plt.show()

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
data["NormalizedAmount"]=ss.fit_transform(data["Amount"].values.reshape(-1,1))
data = data.drop(["Amount"],axis=1)
data = data.drop(["Time"],axis=1)
Y = data[["class"]]
X = data.drop(["class"],axis=1)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3,random_state=0)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
model_dtc = dtc.fit(xtrain,ytrain)
pred = model_dtc.predict(xtest)


from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix
print('Accuracy score is: ',round(accuracy_score(ytest,pred),4))
print("============================================")
print('Classification report: \n',classification_report(ytest,pred))
print("============================================")
print('Confusion Matrix: \n',plot_confusion_matrix(model_dtc,xtest,ytest))


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
model_rfc = rfc.fit(xtrain,ytrain)
pred = model_rfc.predict(xtest)

from sklearn.metrics import accuracy_score,classification_report,plot_confusion_matrix
print('Accuracy score is: ',round(accuracy_score(ytest,pred),4))
print("============================================")
print('Classification report: \n',classification_report(ytest,pred))
print("============================================")
print('Confusion Matrix: \n',plot_confusion_matrix(model_rfc,xtest,ytest))
plt.show()
