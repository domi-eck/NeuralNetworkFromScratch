import numpy as np

class SoftMax:
    def forward(self, input_tensor, label_tensor):
        #input x
        #label y
        #predlabel y
        self.input = input_tensor
        a = np.exp(input_tensor)

        colum_sums = 1/ np.sum(a, 0)
        colum_sums = np.diag(colum_sums)

        y_h = np.dot(a, colum_sums)
        loss = 0
        lossA = label_tensor
        lossA = np.where(lossA == 1, -1 * np.log( y_h ) ,0)


        return np.sum(lossA)
    def backward(self, label_tensor):
        a =2
    def predict(self, input_tensor):
        a = 2
