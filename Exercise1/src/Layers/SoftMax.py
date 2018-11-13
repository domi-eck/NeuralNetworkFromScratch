import numpy as np

class SoftMax:
    def forward(self, input_tensor, label_tensor):

        input_tensor = input_tensor - np.max(input_tensor)
        self.predict(input_tensor)
        a = np.exp(input_tensor)

        colum_row = 1/ np.sum(a, 1)
        colum_row = np.diag(colum_row)

        y_h = np.dot(colum_row, a)
        loss = 0
        lossA = label_tensor
        lossA = np.where(lossA == 1, -1 * np.log( y_h ) ,0)


        return np.sum(lossA)

    def backward(self, label_tensor):

        return np.where(label_tensor == 1, self.y_h -1, self.y_h)

    def predict(self, input_tensor):
        a = np.exp(input_tensor)

        colum_sums = 1/ np.sum(a, 1)
        colum_sums = np.diag(colum_sums)

        self.y_h = np.dot(colum_sums, a)
        return(self.y_h)
