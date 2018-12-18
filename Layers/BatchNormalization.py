import  numpy as np
import Layers.Base as Base

class BatchNormalization(Base.Base):
    def __init__(self, gamma = 1, beta = 1):
        self.gamma = gamma
        self.phase = Base.Phase.train
    def forward(self, input_tensor):
        IsConvInput = False
        if input_tensor.ndim == 4:
            input_tensor = np.transpose(input_tensor, (0, 3, 2, 1))
            shapeBevorReshaping = input_tensor.shape
            input_tensor = input_tensor.reshape(-1, input_tensor.shape[3])
            IsConvInput = True
        if self.phase == Base.Phase.train:
            self.mean = np.mean(input_tensor, axis=0)
            self.var = np.var(input_tensor, axis=0)
            self.forward_output = np.zeros_like(input_tensor)
            for a in np.arange(input_tensor.shape[1]):
                nenner = np.sqrt(  self.var[a]   )
                self.forward_output[:,a] = (input_tensor[:,a] - self.mean[a]) / nenner
        if self.phase == Base.Phase.test:
            self.forward_output = np.zeros_like(input_tensor)
            for a in np.arange(input_tensor.shape[1]):
                nenner = np.sqrt(  self.var[a] )
                self.forward_output[:,a] = (input_tensor[:,a] - self.mean[a]) / nenner
        if IsConvInput == True:
            self.forward_output = self.forward_output.reshape(shapeBevorReshaping)
            self.forward_output = np.transpose(self.forward_output, (0,3,2,1))
        return self.forward_output
    def backward(self, error_tensor):
        b = 2