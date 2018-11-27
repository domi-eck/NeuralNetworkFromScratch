import numpy as np


class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape   = stride_shape
        self.pooling_shape  = pooling_shape
    def forward(self, input_tensor):
        dy = self.stride_shape[0]
        dx = self.stride_shape[1]
        py = self.pooling_shape[0]
        px = self.pooling_shape[1]

        self.batchsize = input_tensor.shape[0]
        self.channelsize = input_tensor.shape[1]
        self.outputY = int(np.floor(input_tensor.shape[2] / self.stride_shape[0]))
        self.outputX = int(np.floor(input_tensor.shape[3] / self.stride_shape[1]))
        while((self.outputY - 1) * self.stride_shape[0] + self.pooling_shape[0] > input_tensor.shape[2]):
            self.outputY -= 1
        while ((self.outputX - 1) * self.stride_shape[1] + self.pooling_shape[1] > input_tensor.shape[3]):
            self.outputX -= 1

        output_tensor = np.zeros([self.batchsize, self.channelsize, self.outputY, self.outputX])



        maxValues = np.zeros(input_tensor.shape)
        #starting at x 0 y 0
        self.mask = np.zeros(input_tensor.shape)
        self.mask[:, :, 0:py, 0:px] += 1

        for yy in np.arange(self.outputY):
            self.maskrolledY = np.roll(self.mask, dy * yy, 2)
            for xx in np.arange(self.outputX):
                maskedRolledXY = np.roll(self.maskrolledY, dx * xx, 3)
                maskedInput = np.multiply(maskedRolledXY, input_tensor)
                #dont know if channel is seperate
                for channel in np.arange(input_tensor.shape[1]):
                    for batch in np.arange(input_tensor.shape[0]):
                        output_tensor[batch, channel, yy, xx] = np.amax(maskedInput[batch, channel])






        return output_tensor
    def backward(self, error_tensor):
        return error_tensor
