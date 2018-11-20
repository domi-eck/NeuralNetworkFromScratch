import numpy as np



class Constant:
    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights = np.ones([fan_in, fan_out])*self.c
        return self.weights



class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        self.weights = np.random.rand(fan_in , fan_out)
        return self.weights


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        self.sigma = np.sqrt(2/(fan_in + fan_out))
        self.weights = np.random.normal(0, self.sigma, [fan_in, fan_out])
        return self.weights


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        self.sigma = np.sqrt(2.0/fan_in)
        self.weights = np.random.normal(0, self.sigma, [fan_in, fan_out])
        return self.weights