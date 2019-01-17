import numpy as np


class Sigmoid:

    def __init__(self):
        self.activations = np.array([])

    def forward(self, input_tensor):
        self.activations = 1/(1 + np.exp(-input_tensor))
        return self.activations

    def backward(self, error_tensor):
        return self.activations*(1-self.activations)*error_tensor
