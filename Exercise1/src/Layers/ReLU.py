import numpy as np

class ReLU:
    def forward(self, input_tensor):
        self.input_ten = input_tensor
        return input_tensor.clip(min=0)
    def backward(self, error_tensor):
        temp = np.where(error_tensor>0, 1, 0)
        temp = np.multiply(temp, error_tensor)
        return temp
