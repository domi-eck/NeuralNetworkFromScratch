import numpy as np



class Constant:
    def __init__(self, constant):
        self.c = constant
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.ones(weights_shape)*self.c
        return weights



class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.rand(np.product(weights_shape))
        weights = weights.reshape(weights_shape)
        return weights


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_in + fan_out))
        weights = np.random.normal(0, sigma, weights_shape)
        return weights


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2.0/fan_in)
        weights = np.random.normal(0, sigma, weights_shape)
        return weights