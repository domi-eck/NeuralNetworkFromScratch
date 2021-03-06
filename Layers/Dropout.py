
import  numpy as np
from Layers import Base


class Dropout(Base.Base):
    def __init__(self, probability):
        self.DropoutPro = probability
        self.phase = Base.Phase.train
    def forward(self, input_tensor):
        if self.phase == Base.Phase.train:
            self.mask = np.random.choice(2,input_tensor.shape, p=[1-self.DropoutPro, self.DropoutPro])
            self.maskedInputTensor = np.multiply(input_tensor, self.mask) * (1/self.DropoutPro)
            return self.maskedInputTensor
        if self.phase == Base.Phase.test:

            return input_tensor
    def backward(self, error_tensor):
        if self.phase == Base.Phase.train:
            return error_tensor * self.mask

