import  numpy as np
import Layers.Base as Base

class BatchNormalization(Base.Base):
    def __init__(self, gamma = 1, beta = 1):
        self.gamma = gamma
        self.phase = Base.Phase.train
    def forward(self, input_tensor):
        if self.phase == Base.Phase.train:
            self.mean = np.mean(input_tensor, axis=0)
            self.var = np.var(input_tensor, axis=0)
            self.forward_output = np.zeros_like(input_tensor)
            for a in np.arange(input_tensor.shape[1]):
                nenner = np.sqrt(  self.var[a]   )
                self.forward_output[:,a] = (input_tensor[:,a] - self.mean[a]) / nenner
        if self.phase == Base.Phase.test:
            print("Mean: " + str( self.mean ) + " Var: " + str( self.var))
            for a in np.arange(input_tensor.shape[1]):
                nenner = np.sqrt(  self.var[a] )
                self.forward_output[:,a] = (input_tensor[:,a] - self.mean[a]) / nenner
        return self.forward_output
    def backward(self, error_tensor):
        b = 2