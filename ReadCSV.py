import pandas as pd

class ReadCSV:
    def __init__(self, path):
        self.path = path

    def readcsv(self):
        return pd.read_csv(self.path)
