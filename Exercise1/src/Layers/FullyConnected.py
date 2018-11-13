import numpy as np

class FullyConnected:
    def __init__(self, input_size, output_size):
        self.W = np.random.rand(input_size, output_size)
        self.delta = 1
        print(self.W)


    def forward(self, input_tensor):
        self.last_input_tensor = input_tensor
        return np.dot(input_tensor, self.W)


    def backward(self, error_tensor):
        self.W = self.W - self.delta * np.dot(self.last_input_tensor.transpose(), error_tensor)
        return np.dot(error_tensor, self.W.transpose())



if __name__ == '__main__':
    dummy = 1
    fc = FullyConnected(3, 2)
    output_tensor = fc.forward(np.random.rand(2, 3))
    print(output_tensor)