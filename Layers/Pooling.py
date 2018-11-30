import numpy as np


class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape   = stride_shape
        self.pooling_shape  = pooling_shape
    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.listOfWinner = list()
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


        self.Winner = np.zeros([*output_tensor.shape, 2])
        self.maxValues = np.zeros(input_tensor.shape)
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
                        max = np.amax(maskedInput[batch, channel])
                        output_tensor[batch, channel, yy, xx] = max
                        i = np.where(input_tensor==max)
                        self.Winner[batch, channel, yy, xx, 0] = i[2]
                        self.Winner[batch, channel, yy, xx, 1] = i[3]

        return output_tensor
    def backward(self, error_tensor):
        dy = self.stride_shape[0]
        dx = self.stride_shape[1]

        dividerMask = np.zeros(self.input_tensor.shape)
        output_tensor = np.zeros(self.input_tensor.shape)




        for yy in np.arange(self.outputY):
            self.maskrolledY = np.roll(self.mask, dy * yy, 2)
            for xx in np.arange(self.outputX):
                maskedRolledXY = np.roll(self.maskrolledY, dx * xx, 3)
                dividerMask += maskedRolledXY
                #dont know if channel is seperat
                for channel in np.arange(self.input_tensor.shape[1]):
                    for batch in np.arange(self.input_tensor.shape[0]):

                        maxYvalue = int( self.Winner[batch, channel, yy, xx, 0] )
                        maxXvalue = int( self.Winner[batch, channel, yy, xx, 1] )
                        output_tensor[batch, channel, maxYvalue,  maxXvalue ] += error_tensor[batch, channel, yy, xx]

                        #maskedRolledXY[batch, channel] *= error_tensor[batch, channel, yy, xx]
                        #maskedRolledXY = np.multiply(maskedRolledXY, self.maxValues)
                        #output_tensor[batch, channel] += maskedRolledXY[batch, channel]
        #Summed, not normed. Idiot
        #output_tensor = np.divide(output_tensor, dividerMask)
        #output_tensor = np.multiply(output_tensor, self.maxValues)

        return output_tensor
