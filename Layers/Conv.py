import numpy as np

class Conv:
    def __init__(self, stride_shape, convulution_shape, num_kernels, learning_rate):
        self.stride_shape = stride_shape
        self.conv_shape = convulution_shape
        self.num_kernels = num_kernels
        self.learning_rate = learning_rate
        self.weights = np.array([])
        self.weights = np.ones(convulution_shape)


    def forward(self, input_tensor):
        #loop for every batch
        for b in np.arange(input_tensor.shape[0]):

            #determine end of indices for convultion
            input_x_size = input_tensor.shape[3]   #starting with 0
            input_y_size = input_tensor.shape[2]   #starting with 0
            x_stride = self.stride_shape[1]
            y_stride = self.stride_shape[0]
            conv_x_size = self.conv_shape[2]
            conv_y_size = self.conv_shape[1]
            # take input tensor for one batch
            one_batch_input_tensor = input_tensor[b]



            #calculate x boarder with stride size of one
            end_x_boarder = input_x_size - conv_x_size
            last_x_index = input_x_size - conv_x_size
            #calculate how the last stride has to end so that no x values are missing in the convolution
            if not end_x_boarder % x_stride is 0:
                last_x_stride = end_x_boarder + (x_stride - end_x_boarder % x_stride)
                # calculate the zeros which has to be padded that the last stride is possible
                zero_x_padding = last_x_stride + conv_x_size - input_x_size
                padding_array = np.zeros([one_batch_input_tensor.shape[0], one_batch_input_tensor.shape[1], 2 ])
                one_batch_input_tensor = np.concatenate([one_batch_input_tensor, padding_array], 2)

            else:
                last_x_stride = end_x_boarder



            #same as for x
            end_y_boarder = input_y_size - conv_y_size
            if not end_y_boarder % y_stride is 0:
                last_y_stride = end_y_boarder + (y_stride - end_y_boarder % y_stride)
                zero_y_padding = last_y_stride + conv_y_size - input_y_size
                padding_array = np.zeros( [one_batch_input_tensor.shape[0], 1, one_batch_input_tensor.shape[2]])
                one_batch_input_tensor = np.concatenate([one_batch_input_tensor, padding_array], 1)
            else:
                last_y_stride = end_y_boarder

            end_y = last_y_stride / y_stride +1 #+1 cause of the zero stride
            end_x = last_x_stride / x_stride +1

            output_tensor = np.zeros([int(end_y), int(end_x)])

            ToDo only one Kernel is used
            ToDo for some reason channel size is not reduced





            # determine end of convulution with respect to kernel size
            #end_y = input_tensor.shape[2] - self.conv_shape[1] + self.stride_shape[1]
            #end_x = input_tensor.shape[3] - self.conv_shape[2] + self.stride_shape[0]

            #calculate output_tensor:
            #output_size_y = end_y/self.stride_shape[0]
            #output_size_x = end_x/self.stride_shape[1]
            #output_tensor = np.zeros([int(output_size_x), int(output_size_y)])

            #loop over every y value with the right stride size
            y_out = 0
            for y in np.arange(0, last_y_stride + y_stride , y_stride):
                #loop over every x value with the right stride size
                x_out = 0
                for x in np.arange(0, last_x_stride + x_stride, x_stride):

                    #get the tensor which will be multiplied with the kernel
                    tensor_for_multiply = one_batch_input_tensor[:, y: (y + conv_y_size), x:(x + conv_x_size)]
                    #make one convulution, first multiply with the weights and the sum over it to get one value
                    output_tensor[y_out][x_out] = np.sum(np.multiply(tensor_for_multiply, self.weights))

                    x_out += 1

                y_out += 1

        return input_tensor


    def backward(self, backward_tensor):
        dummy = 1