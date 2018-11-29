import numpy as np
from scipy import signal as sig

class Conv:
    def __init__(self, stride_shape, convulution_shape, num_kernels, learning_rate):
        self.stride_shape = stride_shape
        self.conv_shape = convulution_shape
        self.num_kernels = num_kernels
        self.learning_rate = learning_rate
        self.weights = np.array([])
        self.weights = np.ones([num_kernels, *convulution_shape])


    def forward(self, input_tensor):
        #loop for every batch
        # determine end of indices for convultion
        input_x_size = input_tensor.shape[3]  # starting with 0
        input_y_size = input_tensor.shape[2]  # starting with 0
        x_stride = self.stride_shape[1]
        y_stride = self.stride_shape[0]
        conv_x_size = self.conv_shape[2]
        conv_y_size = self.conv_shape[1]
        # take input tensor for one batch


        for batch in np.arange(input_tensor.shape[0]):

            # #calculate x boarder with stride size of one
            # end_x_boarder = input_x_size - conv_x_size
            # last_x_index = input_x_size - conv_x_size
            # #calculate how the last stride has to end so that no x values are missing in the convolution
            # if not end_x_boarder % x_stride is 0:
            #     last_x_stride = end_x_boarder + (x_stride - end_x_boarder % x_stride)
            #     # calculate the zeros which has to be padded that the last stride is possible
            #     zero_x_padding = last_x_stride + conv_x_size - input_x_size
            #     padding_array = np.zeros([one_batch_input_tensor.shape[0], one_batch_input_tensor.shape[1], 2 ])
            #     one_batch_input_tensor = np.concatenate([one_batch_input_tensor, padding_array], 2)
            #
            # else:
            #     last_x_stride = end_x_boarder
            #
            #
            #
            # #same as for x
            # end_y_boarder = input_y_size - conv_y_size
            # if not end_y_boarder % y_stride is 0:
            #     last_y_stride = end_y_boarder + (y_stride - end_y_boarder % y_stride)
            #     zero_y_padding = last_y_stride + conv_y_size - input_y_size
            #     padding_array = np.zeros( [one_batch_input_tensor.shape[0], 1, one_batch_input_tensor.shape[2]])
            #     one_batch_input_tensor = np.concatenate([one_batch_input_tensor, padding_array], 1)
            # else:
            #     last_y_stride = end_y_boarder
            #
            # end_y = last_y_stride / y_stride +1 #+1 cause of the zero stride
            # end_x = last_x_stride / x_stride +1
            #
            # output_tensor = np.zeros([int(end_y), int(end_x)])
            #
            # #ToDo only one Kernel is used
            # #ToDo for some reason channel size is not reduced
            # #ToDo do calculation new
            #
            out_x = np.int(np.ceil(input_x_size/x_stride))
            out_y = np.int(np.ceil(input_y_size/y_stride))
            #
            output_tensor = np.zeros([out_y, out_x])
            #
            # padding_y_array = np.zeros([one_batch_input_tensor.shape[0], conv_y_size, one_batch_input_tensor.shape[2]])
            # padding_x_array = np.zeros([one_batch_input_tensor.shape[0], one_batch_input_tensor.shape[1], conv_x_size])
            #
            # one_batch_input_tensor = np.concatenate([one_batch_input_tensor, padding_y_array], 1)
            # one_batch_input_tensor = np.concatenate([one_batch_input_tensor, padding_x_array], 2)

            '''Same Padding:  tries to pad evenly left and right, but if the amount of columns to be added is odd, 
            it will add the extra column to the right, as is the case in this example (the same logic applies 
            vertically: there may be an extra row of zeros at the bottom). In Same padding with stride length one 
            the dimensions stay the same'''

            #Try new Approach, at first only with stride 1
            x_left_padding = np.zeros([input_tensor[batch].shape[0], input_tensor[batch].shape[1],
                                       np.int(np.floor((conv_x_size-1)/2))])
            x_right_padding = np.zeros([input_tensor[batch].shape[0], input_tensor[batch].shape[1],
                                       np.int(np.ceil((conv_x_size-1)/2))])

            padded_input_tensor = np.concatenate([x_left_padding, input_tensor[batch]], 2)
            padded_input_tensor = np.concatenate([padded_input_tensor, x_right_padding], 2)

            y_left_padding = np.zeros([padded_input_tensor.shape[0], np.int(np.floor((conv_y_size-1)/2)),
                                       padded_input_tensor.shape[2]])
            y_right_padding = np.zeros([padded_input_tensor.shape[0], np.int(np.ceil((conv_y_size-1)/2)),
                                        padded_input_tensor.shape[2]])

            padded_input_tensor = np.concatenate([y_left_padding, padded_input_tensor], 1)
            padded_input_tensor = np.concatenate([padded_input_tensor, y_right_padding], 1)



            # determine end of convulution with respect to kernel size
            #end_y = input_tensor.shape[2] - self.conv_shape[1] + self.stride_shape[1]
            #end_x = input_tensor.shape[3] - self.conv_shape[2] + self.stride_shape[0]

            #calculate output_tensor:
            #output_size_y = end_y/self.stride_shape[0]
            #output_size_x = end_x/self.stride_shape[1]
            #output_tensor = np.zeros([int(output_size_x), int(output_size_y)])

            #loop over every y value with the right stride size
            y_out = 0
            for y in np.arange(0, input_y_size, y_stride):
                 #loop over every x value with the right stride size
                 x_out = 0
                 for x in np.arange(0, input_x_size, x_stride):
            #
            #         #get the tensor which will be multiplied with the kernel
                     tensor_for_multiply = padded_input_tensor[:, y: (y + conv_y_size), x:(x + conv_x_size)]
            #         #make one convulution, first multiply with the weights and the sum over it to get one value
                     output_tensor[y_out][x_out] = np.sum(np.multiply(tensor_for_multiply, self.weights))
            #
                     x_out += 1

                 y_out += 1

            #works only with stride 1
            #TODO make with four kernels
            #TODO make for each batch
            output_tensor = sig.convolve(input_tensor[b], self.weights, mode = "same")

        return output_tensor


    def backward(self, backward_tensor):
        dummy = 1