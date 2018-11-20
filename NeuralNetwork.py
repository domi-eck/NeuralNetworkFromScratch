class NeuralNetwork:
    def __init__(self):
        self.loss = list()
        self.layers = list()
        self.data_layer = []
        self.loss_layer = []
        self._didForward = False
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
        while i < iterations:
            self.forward()
            self.backward()
            i += 1
    def test(self, input_tensor):
        nextInput = self.layers[0].forward(input_tensor)
        i = 1
        while i < len(self.layers):
            nextInput = self.layers[i].forward(nextInput)
            i += 1
        return self.loss_layer.predict(nextInput)
