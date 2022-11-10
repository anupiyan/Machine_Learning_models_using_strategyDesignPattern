import pandas as pd

class ReadCSV:
    def __init__(self, path:str):
        self.path = path

    def readcsv(self)->pd.DataFrame:
        return pd.read_csv(self.path)
