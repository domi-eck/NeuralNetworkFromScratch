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
        a = np.exp(input_tensor)
        row_sum = 1 / np.sum(a, 1)
        row_sum = np.diag(row_sum)
        return np.dot(row_sum, a)

