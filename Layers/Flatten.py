import numpy as np

class Flatten:
    def __init__(self):
        self.input_tensor = []
        self.error_tensor = []
        self.input_shape = []
        self.batchSize = []
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.input_shape = input_tensor.shape[1:]
        self.batchSize = input_tensor.shape[0]
        return input_tensor.reshape([self.batchSize, np.prod(self.input_shape)])
    def backward(self, error_tensor):
        self.error_tensor = error_tensor
        return error_tensor.reshape([self.batchSize, *self.input_shape])
