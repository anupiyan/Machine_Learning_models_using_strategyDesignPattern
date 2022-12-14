import pandas as pd


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPipeline:
    def __init__(self, data:pd.DataFrame)->None:
        self.data = data
        self.scale = StandardScaler()
    
    def scalling_data(self, columnname:str)->None:
       self.data["NormalizedAmount"]=self.scale.fit_transform(self.data[columnname].values.reshape(-1,1))

    def drop_column(self, colname:str)->None:
       self.data = self.data.drop([colname],axis=1)

    def data_split_X_Y(self)->pd.DataFrame:
        Y = self.data[["class"]]
        X = self.data.drop(["class"],axis=1)
        return X, Y

    def training_testing_split(self, X:pd.DataFrame, Y:pd.DataFrame)->pd.DataFrame:
        xtrain,xtest,ytrain,ytest = train_test_split(X,Y,test_size=0.3,random_state=0)
        return xtrain, xtest, ytrain, ytest
