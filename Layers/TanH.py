import numpy as np


class TanH:
    def __init__(self):
        self.activations = np.array([])

    def forward(self, input_tensor):
        self.activations = np.tanh(input_tensor)
        return self.activations

    def backward(self, error_tensor):
        return (1 - np.square(self.activations))*error_tensor

