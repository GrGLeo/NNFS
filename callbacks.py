import numpy as np

class EarlyStopping:
    def __init__(self,patience=1,):
        """
        Initialize EarlyStopping with a set value of patience
        """
        self.patience = patience
        self.loss = np.inf
        self.wait = 0
        self.stop = False

    def monitor(self,loss):
        """
        Check if current loss is better than past epoch loss
        else start counting patience
        """

        if loss < self.loss:
            self.loss = loss
            self.wait = 0
        else:
            self.wait += 1

        if self.wait >= self.patience:
            self.stop = True
