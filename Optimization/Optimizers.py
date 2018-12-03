import  numpy as np

class SgdWithMomentum:
    def __init__(self, lr=0.01, momentum=0.0):
        self.learningRate = lr
        self.momentum = momentum

    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        if not hasattr(self, 'vkOld'):
            self.vkOld = - self.learningRate * gradient_tensor
            return weight_tensor + self.vkOld
        self.vkOld = self.vkOld * self.momentum - self.learningRate * gradient_tensor
        return weight_tensor + self.vkOld


class Sgd:
    def __init__(self, lr=0.01):
        self.learningRate = lr

    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        return weight_tensor -  self.learningRate * gradient_tensor

class Adam:
    def __init__(self, lr=0.01, momentum=0.0, roh=0.0):
        self.learningRate = lr
        self.momentum = momentum
        self.roh = roh
        self.iterations = 1

    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        self.gk = gradient_tensor
        if not hasattr(self, 'vk'):
            self.vk = (1 - self.momentum) * self.gk
            self.rk = np.multiply(self.gk, ((1 - self.roh) * self.gk))
        else:
            self.vk = self.momentum * self.vk + (1 - self.momentum) * self.gk
            self.rk = self.roh * self.rk + np.multiply(self.gk, ((1 - self.roh) * self.gk) )
        #bais correction
        self.vkh = self.vk / (1 - np.power(self.momentum, self.iterations))
        self.rkh = self.rk / (1 - np.power(self.roh, self.iterations))
        result = (weight_tensor - self.learningRate * self.vkh / np.sqrt(self.rkh))
        self.iterations += 1
        return result