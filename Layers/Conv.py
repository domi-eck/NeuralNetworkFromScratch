import numpy as np
from Optimization import Optimizers


class Conv:
    def __init__(self, stride_shape, convulution_shape, num_kernels, learning_rate=1):
        self.stride_shape   = stride_shape
        self.conv_shape     = convulution_shape
        self.num_kernels    = num_kernels
        self.learning_rate  = learning_rate
        self.weights        = np.array([])
        self.weights        = np.ones([num_kernels, *convulution_shape])
        self.reshape        = False
        self.bias           = np.ones(num_kernels)
        self.input_tensor   = []

        '''variables which are used for forward and backward'''
        self.input_x_size = 0
        self.input_y_size = 0
        self.input_z_size = 0

        self.x_stride = 0
        self.y_stride = 0
        self.conv_x_size = 0
        self.conv_y_size = 0
        self.batch_num = 0
        self.output_tensor = []
        self.padded_input_tensor = []
        self.out_x = 0
        self.out_y = 0


    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.product(self.weights[0].shape)
        fan_out = np.product([*self.weights[0][0].shape, self.num_kernels])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)

    def get_gradient_weights(self):
        return self.gradient_weights


    def forward(self, input_tensor):

        # if 1D array, add one dimension
        if np.size(input_tensor.shape) is 3:
            input_tensor = np.expand_dims(input_tensor, 3)
            self.stride_shape = np.array([*self.stride_shape, 1])
            self.conv_shape = np.array([*self.conv_shape, 1])
            self.reshape = True


        self.input_x_size = input_tensor.shape[3]  # starting with 0
        self.input_y_size = input_tensor.shape[2]  # starting with 0
        self.input_z_size = input_tensor.shape[1]
        self.x_stride = self.stride_shape[1]
        self.y_stride = self.stride_shape[0]
        self.conv_x_size = self.conv_shape[2]
        self.conv_y_size = self.conv_shape[1]
        self.batch_num = input_tensor.shape[0]


        '''Same Padding: tries to pad evenly left and right, but if the amount of columns to be added is odd, 
        it will add the extra column to the right, as is the case in this example (the same logic applies 
        vertically: there may be an extra row of zeros at the bottom). In Same padding with stride length one 
        the dimensions stay the same'''

        '''Same - Padding is calculated as described above'''
        self.num_x_left_zeros = np.int(np.floor((self.conv_x_size - 1) / 2))
        self.num_x_right_zeros = np.int(np.ceil((self.conv_x_size - 1) / 2))

        self.num_y_left_zeros = np.int(np.floor((self.conv_y_size - 1) / 2))
        self.num_y_right_zeros = np.int(np.ceil((self.conv_y_size - 1) / 2))


        x_left_padding = np.zeros([input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
                                   np.int(np.floor((self.conv_x_size - 1) / 2))])
        x_right_padding = np.zeros([input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
                                    np.int(np.ceil((self.conv_x_size - 1) / 2))])

        self.padded_input_tensor = np.concatenate([x_left_padding, input_tensor], 3)
        self.padded_input_tensor = np.concatenate([self.padded_input_tensor, x_right_padding], 3)

        y_left_padding = np.zeros([self.padded_input_tensor.shape[0], self.padded_input_tensor.shape[1], np.int(np.floor((self.conv_y_size - 1) / 2)),
                                   self.padded_input_tensor.shape[3]])
        y_right_padding = np.zeros([self.padded_input_tensor.shape[0], self.padded_input_tensor.shape[1], np.int(np.ceil((self.conv_y_size - 1) / 2)),
                                    self.padded_input_tensor.shape[3]])

        self.padded_input_tensor = np.concatenate([y_left_padding, self.padded_input_tensor], 2)
        self.padded_input_tensor = np.concatenate([self.padded_input_tensor, y_right_padding], 2)

        '''Calculate and Init the Output Tensor'''
        self.out_x = np.int(np.ceil(self.input_x_size / self.x_stride))
        self.out_y = np.int(np.ceil(self.input_y_size / self.y_stride))

        self.output_tensor = np.zeros([self.batch_num, self.num_kernels, self.out_y, self.out_x])


        for batch in np.arange(input_tensor.shape[0]):
            '''Calculate Convolution for each kernel and each x and y dimension'''
            # loop over every y value with the right stride size
            for kernel in np.arange(self.num_kernels):
                y_out = 0
                for y in np.arange(0, self.input_y_size, self.y_stride):
                    # loop over every x value with the right stride size
                    x_out = 0
                    for x in np.arange(0, self.input_x_size, self.x_stride):
                        # get the tensor which will be multiplied with the kernel
                        tensor_for_multiply = self.padded_input_tensor[batch][:, y: (y + self.conv_y_size),
                                              x:(x + self.conv_x_size)]
                        # make one convolution, first multiply with the weights and the sum over it to get one value
                        self.output_tensor[batch, kernel][y_out][x_out] = np.sum(
                            np.multiply(tensor_for_multiply, self.weights[kernel])) + self.bias[kernel]
                        x_out += 1
                    y_out += 1

        # If there is a One day array remove added dimension which was added in the beginning
        if self.reshape is True:
            self.output_tensor = np.reshape(self.output_tensor,
                                            [self.output_tensor.shape[0], self.output_tensor.shape[1],
                                             self.output_tensor.shape[2]])
            self.reshape = False

        return self.output_tensor

    def backward(self, backward_tensor):
        # if 1D array, add one dimension
        if np.size(backward_tensor.shape) is 3:
            backward_tensor = np.expand_dims(backward_tensor, 3)
            #self.stride_shape = np.array([*self.stride_shape, 1])
            #self.conv_shape = np.array([*self.conv_shape, 1])
            self.weights = np.expand_dims(self.weights, 3)
            self.reshape = True

        '''Calculate gradient'''
        '''Calculate Convolution for each kernel and each x and y dimension'''
        # loop over every y value with the right stride size
        self.gradient_weights = np.zeros([self.num_kernels, self.input_z_size, self.conv_y_size, self.conv_x_size])
        self.error_tensor = np.zeros(self.padded_input_tensor.shape)

        y_out = 0
        for y in np.arange(0, self.input_y_size, self.y_stride):
            # loop over every x value with the right stride size
            x_out = 0
            for x in np.arange(0, self.input_x_size, self.x_stride):

                '''Do The Backward Convolution'''
                tensor_for_multiply = self.padded_input_tensor[:,:, y: (y + self.conv_y_size),
                                      x:(x + self.conv_x_size)]

                for kernel in np.arange(self.num_kernels):
                    for batch in np.arange(self.batch_num):
                        self.gradient_weights[kernel] += \
                            tensor_for_multiply[batch]*backward_tensor[batch, kernel][y_out][x_out]

                        '''calc error tensor'''
                        self.error_tensor[batch][:, y: (y + self.conv_y_size), x:(x + self.conv_x_size)] \
                            += self.weights[kernel]*backward_tensor[batch, kernel][y_out][x_out]

                x_out += 1
            y_out += 1


        y_end = self.error_tensor.shape[2]  - self.num_y_right_zeros
        x_end = self.error_tensor.shape[3]  - self.num_x_right_zeros

        '''Update Kernels'''
        if hasattr(self, 'optimizer'):
            for kernel in np.arange(self.num_kernels):
                self.weights[kernel] = self.optimizer.calculate_update(1, self.weights[kernel], self.gradient_weights[kernel])
        # else:
        #     for kernel in np.arange(self.num_kernels):
        #         self.weights[kernel] -= self.learning_rate*self.gradient_weights[kernel]

        biasGradien = np.zeros_like(self.bias)
        for k in np.arange(backward_tensor.shape[1]):
            biasGradien[k] = np.sum(backward_tensor[:,k,:,:])
        self.bias = biasGradien

        if self.reshape is True:
            self.error_tensor = self.error_tensor[:, :, self.num_y_left_zeros: y_end, self.num_x_left_zeros: x_end]

            self.error_tensor = np.reshape(self.error_tensor,
                                            [self.error_tensor.shape[0], self.error_tensor.shape[1],
                                             self.error_tensor.shape[2]])
            self.weights = np.reshape(self.weights,
                                      [self.weights.shape[0], self.weights.shape[1], self.weights.shape[2]])

            return self.error_tensor

        else:
            return self.error_tensor[:,:, self.num_y_left_zeros: y_end, self.num_x_left_zeros : x_end ]

    def get_gradient_bias(self):
        return self.bias

