import copy
from Layers import  Base
#TODO: refactor Refactor the Neural Network class to add the regularization loss to the data loss. Hint:
#It will be necessary to refactor more classes to get the necessary information. Make use
#of base-classes.


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer
        self.loss = list()
        self.layers = list()
        self.data_layer = []
        self.loss_layer = []
        self._didForward = False
        self.optimizer = optimizer

    def setPhase(self, phase):
        for layer in self.layers:
            layer.phase = phase

    def append_trainable_layer(self, layer):
        layer.set_optimizer(copy.deepcopy(self.optimizer))
        layer.initialize(self.weights_initializer, self.bias_initializer)
        self.layers.append(copy.deepcopy(layer))

    def forward(self):
        self._input_tensor, self._label_tensor = self.data_layer.forward()
        nextInput = self.layers[0].forward(self._input_tensor)
        i = 1
        while i < len(self.layers):
            nextInput = self.layers[i].forward(nextInput)
            i += 1
        self.loss.append( self.loss_layer.forward(nextInput, self._label_tensor) )
        self._didForward = True
        return self.loss[-1]

    def backward(self):
        if(self._didForward):
            nextErrorTensor = self.loss_layer.backward(self._label_tensor)
            i = -1
            while i >= -len(self.layers):
                nextErrorTensor = self.layers[i].backward(nextErrorTensor)
                i -= 1
            self._errorTensor = nextErrorTensor
            return  self._errorTensor

    def train(self, iterations):
        i = 0
        self.setPhase(Base.Phase.train)
        while i < iterations:
            self.forward()
            self.backward()
            i += 1

    def test(self, input_tensor):
        self.setPhase(Base.Phase.test)
        nextInput = self.layers[0].forward(input_tensor)
        i = 1
        while i < len(self.layers):
            nextInput = self.layers[i].forward(nextInput)
            i += 1
        return self.loss_layer.predict(nextInput)
