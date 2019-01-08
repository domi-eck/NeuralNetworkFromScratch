import numpy as np

#TODO: NO use in forward

class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
    def calculate(self, weights):
        #perform subgradient update of weights
        return weights * self.alpha
    def norm(self, weights):
        #TODO: alphas after the norm?
        norm = np.sqrt(np.sum(np.square(weights)))*self.alpha
        return norm

#TODO: sign not used???

class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha
    def calculate(self, weights):
        #perform subgradient update of weights
        return (weights) * self.alpha
    def norm(self, weights):
        a = weights * self.alpha
        absoluts = np.absolute(a)
        norm = np.sum(absoluts)
        return norm