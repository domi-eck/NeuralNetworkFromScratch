import numpy as np

class SoftMax:
    def SoftMax(self):
        self.input
    def forward(self, input_tensor, label_tensor):
        #for numerical stability
        self.input = input_tensor - np.max(input_tensor)

        self.y_h = self.predict(self.input)

        lossA = label_tensor
        lossA = np.where(lossA == 1, -np.log( self.y_h ),0)

        return np.sum(lossA)

    def backward(self, label_tensor):
        return  np.where(label_tensor == 1, self.y_h - 1, self.y_h)

    def predict(self, input_tensor):
        p = np.array(input_tensor)
        a = np.array(input_tensor)
        for i in np.arange(input_tensor.shape[0]):
            a[i] = np.exp(input_tensor[i] - np.max(input_tensor[i]))
            p[i] = a[i]/np.sum(a[i])
        return p

