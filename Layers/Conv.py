import numpy as np
import scipy.signal
import copy

import math
import copy

from Layers import  Base
from Optimization import Optimizers


class Conv:

    # def __init__(self, input_image_shape, stride_shape, convolution_shape, num_kernels):
    #
    #     # z, y, x order
    #     # Check depth S (z) of kernel and image identical (11)
    #     if input_image_shape[0] != convolution_shape[0]:
    #         raise ValueError("Kernel and image have to have the same depth!")
    #         pass
    #     self.input_image_shape = input_image_shape
    #     self.stride_shape = stride_shape
    #     self.convolution_shape = convolution_shape
    #     self.num_kernels = num_kernels  # Equals H <-> output depth
    #
    #     # Learning rate
    #     self.delta = 1
    #
    #     # Initialize the parameters of this layer uniformly random in the range [0; 1)
    #     self.weights = np.random.uniform(0, 1, ((num_kernels,) + convolution_shape))
    #     self.bias = np.random.uniform(0, 1, num_kernels)
    #
    #     self.grad_wrt_weights = None
    #     self.grad_wrt_bias = None
    #
    #     self.optimizer_weights = None
    #     self.optimizer_bias = None
    #
    #     self.output_shape = None

    def __init__(self, stride_shape, convolution_shape, num_kernels, learning_rate=1):
        self.stride_shape   = stride_shape
        self.conv_shape     = convolution_shape
        self.num_kernels    = num_kernels
        self.learning_rate  = learning_rate
        self.weights        = np.array([])
        self.weights        = np.ones([num_kernels, *convolution_shape])
        self.reshape        = False
        self.bias           = np.ones(num_kernels)
        self.input_tensor   = []

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

        self.input_image_shape = np.array([])


        #from pascal
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels  # Equals H <-> output depth


        #end pascal

        self.out_y = 0
        self.hasOptimizer = False


    def set_optimizer(self, optimizer):
        self.hasOptimizer = True
        self.optimizer = copy.deepcopy(optimizer)
        self.biasOptimizer = copy.deepcopy(optimizer)


    def getLoss(self):
        if self.hasOptimizer:
            loss = self.optimizer.getLoss()
            loss += self.biasOptimizer.getLoss()
            return loss
        return 0

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.product(self.weights[0].shape)
        fan_out = np.product([*self.weights[0][0].shape, self.num_kernels])
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, self.num_kernels, 1)

    def get_gradient_weights(self):
        return self.gradient_weights


    def forward(self, input_tensor):
        # if 1D array, add one dimension
        if np.size(input_tensor.shape) is 3:
            input_tensor = np.expand_dims(input_tensor, 3)
            self.weights = np.expand_dims(self.weights, 3)
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
        # '''Calculate and Init the Output Tensor'''
        self.out_x = np.int(np.ceil(self.input_x_size / self.x_stride))
        self.out_y = np.int(np.ceil(self.input_y_size / self.y_stride))
        #
        self.output_tensor = np.zeros([self.batch_num, self.num_kernels, self.out_y, self.out_x])
        self.input_image_shape = input_tensor[0].shape

        '''from pascal'''
        batch_size = input_tensor.shape[0]

        corr_shape = self.input_image_shape[1:]

        # TODO: Move to init ??
        # Compute indices for a later subsampling to perform striding
        # Example: Stride = 4: Keep every 4th row (idx: 0,4,8,...), delete the others
        self.del_stride = []
        corr_strided_shape = ()
        for stride_dim in range(len(self.stride_shape)):
            self.del_stride.append([])
            for i in range(corr_shape[stride_dim]):
                if i % self.stride_shape[stride_dim] != 0:
                    self.del_stride[stride_dim].append(i)
            corr_strided_shape = corr_strided_shape + ((corr_shape[stride_dim] - len(self.del_stride[stride_dim])),)

        # corr_strided_shape = tuple(np.subtract(corr_shape, (len(self.del_stride_row), len(self.del_stride_col))))

        batch_shape = ((self.num_kernels,) + corr_strided_shape)
        res_shape = (batch_size, self.num_kernels * np.prod(corr_strided_shape))
        self.output_shape = corr_strided_shape

        res = np.ndarray(res_shape)

        # Iterate over batches
        for batch_idx in range(batch_size):
            # Check if input_tensor size fits given input_image_shape
            if input_tensor[batch_idx].shape[0] != np.prod(self.input_image_shape) and \
                    input_tensor[batch_idx].shape != self.input_image_shape:
                raise ValueError("input_image_shape does not fit input_tensor shape!")

            # Restore the shape of the input vector
            X = input_tensor[batch_idx].reshape(self.input_image_shape)

            # Result for the current batch element
            res_batch = np.ndarray(batch_shape)

            # Iterate over the number of kernels
            for kernel_idx in range(self.num_kernels):
                # Convolve (correlate)
                cor = scipy.signal.correlate(X, self.weights[kernel_idx], mode='same')

                # Delete zero padding in channel direction
                if cor.shape[0] != 1:
                    corr_clean = cor[1]
                else:
                    corr_clean = cor

                # Add bias
                corr_clean += self.bias[kernel_idx]

                # Delete elements (stride subsampling)
                for stride_dim in range(len(self.stride_shape)):
                    corr_clean = np.delete(corr_clean, self.del_stride[stride_dim], stride_dim)

                res_batch[kernel_idx] = corr_clean
                pass

            pass

            # Flatten and append to result
            self.output_tensor[batch_idx] = res_batch

            pass
        pass

        # Save corrent input for backward pass
        self.current_input_tensor = input_tensor


        # '''end pascal'''
        #
        #
        #
        # '''Same Padding: tries to pad evenly left and right, but if the amount of columns to be added is odd,
        # it will add the extra column to the right, as is the case in this example (the same logic applies
        # vertically: there may be an extra row of zeros at the bottom). In Same padding with stride length one
        # the dimensions stay the same'''
        #
        # '''Same - Padding is calculated as described above'''
        # self.num_x_left_zeros = np.int(np.floor((self.conv_x_size - 1) / 2))
        # self.num_x_right_zeros = np.int(np.ceil((self.conv_x_size - 1) / 2))
        #
        # self.num_y_left_zeros = np.int(np.floor((self.conv_y_size - 1) / 2))
        # self.num_y_right_zeros = np.int(np.ceil((self.conv_y_size - 1) / 2))
        #
        #
        # x_left_padding = np.zeros([input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
        #                            np.int(np.floor((self.conv_x_size - 1) / 2))])
        # x_right_padding = np.zeros([input_tensor.shape[0], input_tensor.shape[1], input_tensor.shape[2],
        #                             np.int(np.ceil((self.conv_x_size - 1) / 2))])
        #
        # self.padded_input_tensor = np.concatenate([x_left_padding, input_tensor], 3)
        # self.padded_input_tensor = np.concatenate([self.padded_input_tensor, x_right_padding], 3)
        #
        # y_left_padding = np.zeros([self.padded_input_tensor.shape[0], self.padded_input_tensor.shape[1], np.int(np.floor((self.conv_y_size - 1) / 2)),
        #                            self.padded_input_tensor.shape[3]])
        # y_right_padding = np.zeros([self.padded_input_tensor.shape[0], self.padded_input_tensor.shape[1], np.int(np.ceil((self.conv_y_size - 1) / 2)),
        #                             self.padded_input_tensor.shape[3]])
        #
        # self.padded_input_tensor = np.concatenate([y_left_padding, self.padded_input_tensor], 2)
        # self.padded_input_tensor = np.concatenate([self.padded_input_tensor, y_right_padding], 2)
        #
        # '''Calculate and Init the Output Tensor'''
        # self.out_x = np.int(np.ceil(self.input_x_size / self.x_stride))
        # self.out_y = np.int(np.ceil(self.input_y_size / self.y_stride))
        #
        # self.output_tensor = np.zeros([self.batch_num, self.num_kernels, self.out_y, self.out_x])
        #
        #
        # for batch in np.arange(input_tensor.shape[0]):
        #     '''Calculate Convolution for each kernel and each x and y dimension'''
        #     # loop over every y value with the right stride size
        #     for kernel in np.arange(self.num_kernels):
        #         y_out = 0
        #         for y in np.arange(0, self.input_y_size, self.y_stride):
        #             # loop over every x value with the right stride size
        #             x_out = 0
        #             for x in np.arange(0, self.input_x_size, self.x_stride):
        #                 # get the tensor which will be multiplied with the kernel
        #                 tensor_for_multiply = self.padded_input_tensor[batch][:, y: (y + self.conv_y_size),
        #                                       x:(x + self.conv_x_size)]
        #                 # make one convolution, first multiply with the weights and the sum over it to get one value
        #                 self.output_tensor[batch, kernel][y_out][x_out] = np.sum(
        #                     np.multiply(tensor_for_multiply, self.weights[kernel]))
        #                 x_out += 1
        #             y_out += 1
        #
        #         self.output_tensor[batch, kernel] += self.bias[kernel]

        # If there is a One day array remove added dimension which was added in the beginning

        if self.reshape is True:
            self.output_tensor = np.reshape(self.output_tensor,
                                            [self.output_tensor.shape[0], self.output_tensor.shape[1],
                                             self.output_tensor.shape[2]])
            self.weights = self.weights[:,:,:, 0]
            self.reshape = False

        return self.output_tensor

    def backward(self, backward_tensor):
        error_tensor = backward_tensor
        self.error_tensor = error_tensor
        # if 1D array, add one dimension
        if np.size(backward_tensor.shape) is 3:
            backward_tensor = np.expand_dims(backward_tensor, 3)
            self.weights = np.expand_dims(self.weights, 3)
            #self.stride_shape = np.array([*self.stride_shape, 1])
            #self.conv_shape = np.array([*self.conv_shape, 1])
            #self.weights = np.expand_dims(self.weights, 3)
            self.reshape = True

        '''Calculate and Init the Output Tensor'''
        self.out_x = np.int(np.ceil(self.input_x_size / self.x_stride))
        self.out_y = np.int(np.ceil(self.input_y_size / self.y_stride))

        self.output_tensor = np.array([])

        batch_size = error_tensor.shape[0]

        # Gradient w.r.t bias for all batch elements
        grad_wrt_bias_batch = np.zeros((batch_size, self.num_kernels))
        grad_wrt_weights_batch = np.zeros((batch_size,) + self.weights.shape)
        res_batch = np.zeros((batch_size,) + (np.prod(self.input_image_shape),))

        # TODO: Move to init ??
        # Create a list of indices where zero columns / rows should be inserted for upsampling (stride)
        ins_stride = []
        E_n_upsampled_shape = ()
        # Iterate over dimensions
        # for stride_dim in range(len(self.stride_shape)):
        #     ins_stride.append([])
        #     for i in range(1, self.output_shape[stride_dim]):
        #         for j in range(self.stride_shape[stride_dim] - 1):
        #             ins_stride[stride_dim].append(i)
        for stride_dim in range(len(self.stride_shape)):
            ins_stride.append([])
            i = 1
            while len(ins_stride[stride_dim]) < len(self.del_stride[stride_dim]):
                # for i in range(1, self.output_shape[stride_dim]):
                for j in range(self.stride_shape[stride_dim] - 1):
                    ins_stride[stride_dim].append(i)
                i += 1

        # Iterate over the elements of the batch
        for batch_idx in range(batch_size):
            pass

            # Reshape the error tensor of the current batch element
            E_n = error_tensor[batch_idx].reshape((self.num_kernels,) + self.output_shape)

            # Compute the gradient w.r.t. bias by summing up the elements of every channel
            grad_wrt_bias_batch[batch_idx] = np.sum(E_n, axis=tuple(range(1, len(self.output_shape) + 1)))

            # Perform upsampling of the error tensor (because of stride) (add rows and cols with value zero)
            E_n_upsampled = E_n.copy()
            # Iterate over the dimensions
            for stride_dim in range(len(self.output_shape)):
                E_n_upsampled = np.insert(E_n_upsampled, obj=ins_stride[stride_dim], values=0, axis=stride_dim + 1)

            # Compute the gradient w.r.t. weights (17) -----------------------------------------------
            # Reshape and pad the input tensor with half of the kernel size
            X_pad = self.current_input_tensor[batch_idx].reshape(self.input_image_shape)
            for dim in range(len(self.input_image_shape) - 1):
                # Padding:  For odd kernel shape: E.g. 9 -> Pad 4 on each side
                #           For even kernel shape: E.g. 8 -> Pad 4 at the beginning and 3 at the end
                ext_width_begin = math.floor(self.weights.shape[dim - 2] / 2)
                ext_width_end = math.floor((self.weights.shape[dim - 2] - 1) / 2)
                ind_begin = [0] * ext_width_begin
                ind_end = [self.input_image_shape[dim + 1] + ext_width_begin] * ext_width_end
                X_pad = np.insert(arr=X_pad, obj=ind_begin, values=0, axis=dim + 1)
                X_pad = np.insert(arr=X_pad, obj=ind_end, values=0, axis=dim + 1)

            # Perform convolution for each kernel
            for kernel_idx in range(self.num_kernels):
                grad_wrt_weights_batch[batch_idx][kernel_idx] = scipy.signal.correlate(X_pad, np.expand_dims(
                    E_n_upsampled[kernel_idx], axis=0), mode='valid')

            # Error tensor of the next layer <-> Gradient w.r.t. lower layers (15) ---------------------------
            # Reslice the kernels (15)
            # TODO: Check
            # TODO: Use np.swapaxes???
            tmp = list(self.weights.shape)
            tmp[0], tmp[1] = tmp[1], tmp[0]
            weights_resliced_shape = tuple(tmp)
            weights_resliced = np.zeros(weights_resliced_shape)
            for new_kernel_idx in range(weights_resliced.shape[0]):
                # Flip new kernels!
                for slice_idx, ee in zip(reversed(range(weights_resliced.shape[1])), range(weights_resliced.shape[1])):
                    # weights_resliced[new_kernel_idx][slice_idx] = self.weights[slice_idx][new_kernel_idx]
                    weights_resliced[new_kernel_idx][ee] = self.weights[slice_idx][new_kernel_idx]
                    pass

            # Convolve the error tensor with the resliced kernels
            # Res should have Input image shape
            E_res = np.zeros(self.input_image_shape)
            for channel in range(weights_resliced.shape[0]):
                conv = scipy.signal.convolve(weights_resliced[channel], E_n_upsampled, mode='full')
                shape_diff = np.subtract(conv.shape, self.input_image_shape)
                # TODO: Floor or ceil??
                start_idx = tuple(np.floor(shape_diff / 2).astype(int))
                end_idx = tuple(np.floor(conv.shape - (shape_diff / 2)).astype(int))
                # conv_clean = conv[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1], start_idx[2]:end_idx[2]]
                if len(conv.shape) == 2:
                    E_res[channel] = conv[math.floor(conv.shape[0] / 2), start_idx[1]:end_idx[1]]
                elif len(conv.shape) == 3:
                    E_res[channel] = conv[math.floor(conv.shape[0] / 2), start_idx[1]:end_idx[1],
                                     start_idx[2]:end_idx[2]]
                else:
                    raise NotImplementedError
            if batch_idx == 0:
                self.output_tensor = np.array([E_res])
            else:
                self.output_tensor = np.append(self.output_tensor, np.array([E_res]), 0)








        # '''Calculate gradient'''
        # '''Calculate Convolution for each kernel and each x and y dimension'''
        # # loop over every y value with the right stride size
        # self.gradient_weights = np.zeros([self.num_kernels, self.input_z_size, self.conv_y_size, self.conv_x_size])
        # self.error_tensor = np.zeros(self.padded_input_tensor.shape)
        #
        # y_out = 0
        # for y in np.arange(0, self.input_y_size, self.y_stride):
        #     # loop over every x value with the right stride size
        #     x_out = 0
        #     for x in np.arange(0, self.input_x_size, self.x_stride):
        #
        #         '''Do The Backward Convolution'''
        #         tensor_for_multiply = self.padded_input_tensor[:,:, y: (y + self.conv_y_size),
        #                               x:(x + self.conv_x_size)]
        #
        #         for kernel in np.arange(self.num_kernels):
        #             for batch in np.arange(self.batch_num):
        #                 self.gradient_weights[kernel] += \
        #                     tensor_for_multiply[batch]*backward_tensor[batch, kernel][y_out][x_out]
        #                 if(self.weights[kernel]*backward_tensor[batch, kernel][y_out][x_out]).shape != (self.error_tensor[batch][:, y: (y + self.conv_y_size), x:(x + self.conv_x_size)]).shape:
        #                     dummy = 1
        #                 '''calc error tensor'''
        #                 self.error_tensor[batch][:, y: (y + self.conv_y_size), x:(x + self.conv_x_size)] \
        #                     += self.weights[kernel]*backward_tensor[batch, kernel][y_out][x_out]
        #
        #         x_out += 1
        #     y_out += 1
        #
        #
        # y_end = self.error_tensor.shape[2]  - self.num_y_right_zeros
        # x_end = self.error_tensor.shape[3]  - self.num_x_right_zeros
        #
        #'''Update Kernels'''
        # if hasattr(self, 'optimizer'):
        #     for kernel in np.arange(self.num_kernels):
        #         self.weights[kernel] = self.optimizer.calculate_update(self.learning_rate, self.weights[kernel], self.gradient_weights[kernel])
        #     biasGradien = np.zeros_like(self.bias)
        #     for k in np.arange(backward_tensor.shape[1]):
        #         biasGradien[k] = np.sum(backward_tensor[:, k, :, :])
        #     self.bias = self.biasOptimizer.calculate_update(self.learning_rate, self.bias, biasGradien)
        # else:
        #      for kernel in np.arange(self.num_kernels):
        #          self.weights[kernel] -= self.learning_rate*self.gradient_weights[kernel]
        #
        #      self.biasGradien = np.zeros_like(self.bias)
        #      for k in np.arange(backward_tensor.shape[1]):
        #          self.biasGradien[k] = np.sum(backward_tensor[:, k, :, :])
        #      self.bias = self.bias - self.learning_rate*self.biasGradien


        # Gradients: Finally sum up over the batches
        self.grad_wrt_bias = np.sum(grad_wrt_bias_batch, axis=0)
        self.grad_wrt_weights = np.sum(grad_wrt_weights_batch, axis=0)
        self.gradient_weights = self.grad_wrt_weights
        self.biasGradien = self.grad_wrt_bias


        # Use optimizer to update weights and bias
        # TODO: Only if set? Default optimizer?
        if self.hasOptimizer:
            self.weights = self.optimizer.calculate_update(self.learning_rate, self.weights, self.grad_wrt_weights)
            self.bias = self.biasOptimizer.calculate_update(self.learning_rate, self.bias, self.grad_wrt_bias)



        if self.reshape is True:
            #self.weights = self.weights[:, :, :, 0]
            self.output_tensor = self.output_tensor[:, :, :, 0]
            return self.output_tensor
        else:

            return self.output_tensor

    def get_gradient_bias(self):
        return self.biasGradien


