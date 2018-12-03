import numpy as np



class Constant:
    def initialize(self, weights_shape, fan_in, fan_out):
        c = 0.1 #value of constant
        self.weights = np.ones([fan_in, fan_out])*c
        return self.weights



class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.rand(fan_in , fan_out)
        weights = weights.reshape(weights_shape)
        return weights


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_in + fan_out))
        weights = np.random.normal(0, sigma, [fan_in, fan_out])
        weights = weights.reshape(weights_shape)
        return weights


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2.0/fan_in)
        weights = np.random.normal(0, sigma, [fan_in, fan_out])
        weights = weights.reshape(weights_shape)
        return weights