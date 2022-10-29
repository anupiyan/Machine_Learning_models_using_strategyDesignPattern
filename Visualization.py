
import seaborn as sb
import matplotlib.pyplot as plt

class visualization:
    def __init__(self, data):
        self.data = data

    def correlationship_vs(self):
        corr = data.corr()
        ax = sb.heatmap(corr, annot=True, cmap="YlGnBu")
        ## Display the visualization of the Confusion Matrix.
        plt.show()

