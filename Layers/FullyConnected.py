import numpy as np
from Layers import Base
from Optimization import Optimizers

class FullyConnected:
    def __init__(self, input_size, output_size, dumm = 0):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size +1, output_size)
        self.delta = dumm
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def forward(self, input_tensor):
        b = np.ones([np.size(input_tensor, 0), 1])

        input_tensor = np.hstack([input_tensor, b])
        self.last_input_tensor = input_tensor


        return np.dot(input_tensor, self.weights)


    def backward(self, error_tensor):

        self.gradient = np.dot(self.last_input_tensor.transpose(), error_tensor)
        #now with optimizer#
        #check if optimizer is used
        if hasattr(self, 'optimizer'):
            self.weights = self.optimizer.calculate_update(self.delta, self.weights, self.gradient)
        else:
            self.weights = self.weights - self.delta * self.gradient
        #end of with optimizer#
        #removing the bais
        self.back_output = self.weights[0:np.size(self.weights, 0) - 1, :]
        return np.dot(error_tensor, self.back_output.transpose())

    def get_gradient_weights(self):
        return self.gradient

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = weights_initializer.initialize(np.array([self.input_size, self.output_size]), self.input_size,
                                                      self.output_size)
        self.bias = bias_initializer.initialize([1, self.output_size], 1, self.output_size)
        self.weights = np.vstack([self.weights, self.bias])


if __name__ == '__main__':
    dummy = 1
    fc = FullyConnected(3, 2)
    output_tensor = fc.forward(np.random.rand(2, 3))
    print(output_tensor)