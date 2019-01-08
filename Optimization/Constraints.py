import numpy as np

#TODO: NO use in forward

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
    def calculate(self, weights):
        #perform subgradient update of weights
        return weights * self.alpha
    def norm(self, weights):
        square = np.square(weights)
        norm = np.sum(square)
        norm = np.sqrt(norm) * self.alpha
        return norm


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
    def calculate(self, weights):
        #perform subgradient update of weights
        return np.sign( weights ) * self.alpha
    def norm(self, weights):
        absoluts = np.absolute(weights)
        norm = np.sum(absoluts)* self.alpha
        return norm