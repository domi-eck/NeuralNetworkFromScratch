import  numpy as np
import Layers.Base as Base

class BatchNormalization(Base.Base):
    def __init__(self, gamma = 1, beta = 1):
        self.gamma = gamma
        self.mean = beta
        self.var = 1
        self.phase = Base.Phase.train
    def forward(self, input_tensor):
        if self.phase == Base.Phase.train:
            self.forward_output = np.zeros_like(input_tensor)
            for a in np.arange(input_tensor.shape[1]):
                nenner = np.sqrt(  np.var(input_tensor[:,a] ) + 0.00000001 )
                self.forward_output[:,a] = (input_tensor[:,a] - np.mean(input_tensor[:,a])) / nenner
                self.mean = np.mean(input_tensor[:,a])
                self.var = np.var(input_tensor[:,a] )
        if self.phase == Base.Phase.test:
            nenner = np.sqrt(self.var + 0.00000001)
            self.forward_output = (input_tensor - self.mean) / nenner
        return self.forward_output
    def backward(self, error_tensor):
        b = 2