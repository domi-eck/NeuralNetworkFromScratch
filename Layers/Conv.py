import numpy as np

class Conv:
    def __init__(self, stride_shape, convulution_shape, num_kernels, learning_rate):
        self.stride_shape = stride_shape
        self.conv_shape = convulution_shape
        self.num_kernels = num_kernels
        self.learning_rate = learning_rate
        self.weights = np.array([])
        self.weights = np.ones([num_kernels, *convulution_shape])
        self.reshape = False
        self.bias = np.ones(num_kernels)

    def forward(self, input_tensor):

        #if 1D array, add one dimension
        if np.size(input_tensor.shape) is 3:
            input_tensor = np.expand_dims(input_tensor, 3)
            self.stride_shape = np.array([*self.stride_shape, 1])
            self.conv_shape = np.array([*self.conv_shape, 1])
            self.reshape = True

        input_x_size = input_tensor.shape[3]  # starting with 0
        input_y_size = input_tensor.shape[2]  # starting with 0
        x_stride = self.stride_shape[1]
        y_stride = self.stride_shape[0]
        conv_x_size = self.conv_shape[2]
        conv_y_size = self.conv_shape[1]
        batch_num = input_tensor.shape[0]

        for batch in np.arange(input_tensor.shape[0]):

            '''Calculate and Init the Output Tensor'''
            out_x = np.int(np.ceil(input_x_size / x_stride))
            out_y = np.int(np.ceil(input_y_size / y_stride))

            output_tensor = np.zeros([batch_num, self.num_kernels, out_y, out_x])

            '''Same Padding: tries to pad evenly left and right, but if the amount of columns to be added is odd, 
            it will add the extra column to the right, as is the case in this example (the same logic applies 
            vertically: there may be an extra row of zeros at the bottom). In Same padding with stride length one 
            the dimensions stay the same'''

            '''Same - Padding is calculated as described above'''
            x_left_padding = np.zeros([input_tensor[batch].shape[0], input_tensor[batch].shape[1],
                                       np.int(np.floor((conv_x_size - 1) / 2))])
            x_right_padding = np.zeros([input_tensor[batch].shape[0], input_tensor[batch].shape[1],
                                        np.int(np.ceil((conv_x_size - 1) / 2))])

            padded_input_tensor = np.concatenate([x_left_padding, input_tensor[batch]], 2)
            padded_input_tensor = np.concatenate([padded_input_tensor, x_right_padding], 2)

            y_left_padding = np.zeros([padded_input_tensor.shape[0], np.int(np.floor((conv_y_size - 1) / 2)),
                                       padded_input_tensor.shape[2]])
            y_right_padding = np.zeros([padded_input_tensor.shape[0], np.int(np.ceil((conv_y_size - 1) / 2)),
                                        padded_input_tensor.shape[2]])

            padded_input_tensor = np.concatenate([y_left_padding, padded_input_tensor], 1)
            padded_input_tensor = np.concatenate([padded_input_tensor, y_right_padding], 1)

            '''Calculate Convolution for each kernel and each x and y dimension'''
            # loop over every y value with the right stride size
            for kernel in np.arange(self.num_kernels):
                y_out = 0
                for y in np.arange(0, input_y_size, y_stride):
                    # loop over every x value with the right stride size
                    x_out = 0
                    for x in np.arange(0, input_x_size, x_stride):
                        # get the tensor which will be multiplied with the kernel
                        tensor_for_multiply = padded_input_tensor[:, y: (y + conv_y_size), x:(x + conv_x_size)]
                        # make one convolution, first multiply with the weights and the sum over it to get one value
                        output_tensor[batch, kernel][y_out][x_out] = np.sum(
                            np.multiply(tensor_for_multiply, self.weights[kernel])) + self.bias[kernel]
                        x_out += 1
                    y_out += 1

        #If there is a One day array remove added dimension which was added in the beginning
        if self.reshape is True:
            output_tensor = np.reshape(output_tensor, [output_tensor.shape[0], output_tensor.shape[1],
                                       output_tensor.shape[2]])
        return output_tensor

    def backward(self, backward_tensor):
        dummy = 1
