import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size):
        self.weights = np.random.rand(input_size, output_size)
        self.delta = 1
        print(self.weights)


    def forward(self, input_tensor):
        self.last_input_tensor = input_tensor
        return np.dot(input_tensor, self.weights)


    def backward(self, error_tensor):

        self.gradient = np.dot(self.last_input_tensor.transpose(), error_tensor)
        self.weights = self.weights - self.delta * self.gradient

        return np.dot(error_tensor, self.weights.transpose())

    def get_gradient_weights(self):
        return self.gradient


if __name__ == '__main__':
    dummy = 1
    fc = FullyConnected(3, 2)
    output_tensor = fc.forward(np.random.rand(2, 3))
    print(output_tensor)