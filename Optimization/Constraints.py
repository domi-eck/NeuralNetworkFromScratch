import numpy as np

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
    def calculate(self, weights):
        #perform subgradient update of weights
        return weights * self.alpha
    def norm(self, weights):
        a = weights * self.alpha
        square = np.square(a)
        norm = np.sum(square)
        norm = np.sqrt(norm)
        return norm

class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
    def calculate(self, weights):
        #perform subgradient update of weights
        return weights * self.alpha
    def norm(self, weights):
        a = weights * self.alpha
        absoluts = np.absolute(a)
        norm = np.sum(absoluts)
        return norm