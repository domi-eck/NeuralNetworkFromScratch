import  numpy as np

class HasRegularizer:
    def __init__(self):
        self.hasRegularizer = False
    def add_regularizer(self, regularizer):
        self.regularizer = regularizer
        self.hasRegularizer = True

class SgdWithMomentum (HasRegularizer):
    def __init__(self, lr=0.01, momentum=0.0):
        self.learningRate = lr
        self.momentum = momentum
        HasRegularizer.__init__(self)
    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        constrain = 0
        if(self.hasRegularizer):
            constrain = self.regularizer.calculate(weight_tensor) * self.learningRate * individual_delta
        if not hasattr(self, 'vkOld'):
            self.vkOld = - self.learningRate * gradient_tensor
            return weight_tensor + self.vkOld
        self.vkOld = self.vkOld * self.momentum - self.learningRate * gradient_tensor
        return weight_tensor + self.vkOld - constrain


class Sgd (HasRegularizer):
    def __init__(self, lr=0.01):
        self.learningRate = lr
        HasRegularizer.__init__(self)
    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        constrain = 0
        if(self.hasRegularizer):
            constrain = self.regularizer.calculate(weight_tensor) * self.learningRate * individual_delta
        return weight_tensor - individual_delta *  self.learningRate * gradient_tensor - constrain

class Adam (HasRegularizer):
    def __init__(self, lr=0.01, momentum=0.0, roh=0.0):
        self.learningRate = lr
        self.momentum = momentum
        self.roh = roh
        self.iterations = 1
        HasRegularizer.__init__(self)
    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        constrain = 0
        self.gk = gradient_tensor
        if(self.hasRegularizer):
            constrain = self.regularizer.calculate(weight_tensor) * self.learningRate * individual_delta
        if not hasattr(self, 'vk'):
            self.vk = (1 - self.momentum) * self.gk
            self.rk = np.multiply(self.gk, ((1 - self.roh) * self.gk))
        else:
            self.vk = self.momentum * self.vk + (1 - self.momentum) * self.gk
            self.rk = self.roh * self.rk + np.multiply(self.gk, ((1 - self.roh) * self.gk) )
        #bais correction
        self.vkh = self.vk / (1 - np.power(self.momentum, self.iterations))
        self.rkh = self.rk / (1 - np.power(self.roh, self.iterations))
        result = (weight_tensor - self.learningRate * (self.vkh + 0.0001)/ (np.sqrt(self.rkh) + 0.0001))
        self.iterations += 1
        return result - constrain

