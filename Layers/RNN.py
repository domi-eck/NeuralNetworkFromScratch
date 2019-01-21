import numpy as np
from Layers import Sigmoid
from Layers import TanH


class RNN:
    def __init__(self, input_size, hidden_size, output_size, bptt_length):
        """
        Recurrent Neuronal Network
        :param input_size: the dimension of the input vector
        :param hidden_size: dimension of the hidden state
        :param output_size:
        :param bptt_length: controls how many steps backwards are considered in the calculation of the gradient
                            with respect to the weights
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bptt_length = bptt_length  # (batch size bzw. time dimension)

        self.hidden_state = np.zeros(hidden_size)
        self.same_sequence = False

        # init of weights for hidden state
        self.whh = np.random.rand(hidden_size, hidden_size)
        self.wxh = np.random.rand(input_size, hidden_size)
        self.why = np.random.rand(hidden_size, output_size)
        self.bh = np.ones(hidden_size)
        self.by = np.ones(output_size)

        """ Activations and weights of every time step are needed for the backward pass
         therefore the activations are stored in the objects of Sigmoid and TanH class, 
         there is a instance for every time step: """
        # init Sigmoid functions, also to store them for the backward algorithm
        self.list_sigmoid = [Sigmoid.Sigmoid()] * self.bptt_length
        # init Tanh functions, also to store them for the backward algorithm
        self.list_tanh = [TanH.TanH()] * self.bptt_length
        # also the init of the TanH has to be stored
        # init input for tanh function, store every step for backward
        self.u = [np.array([])] * self.bptt_length

        # init output with zeros
        self.output = np.zeros([self.bptt_length, self.output_size])

    def toggle_memory(self):
        """
        switches a boolean member variable "same_sequence" representing whether the RNN regards subsequent sequences as
        a belonging to the same long sequence
        """
        if self.same_sequence is False:
            self.same_sequence = True
        else:
            self.same_sequence = False

    def forward(self, input_tensor):
        """
        Consider the batch dimension as the time dimension of a sequence over which the recurrence is performed.
        The first hidden state for this iteration is all zero if same_sequence is False,
        otherwise restore the hidden state from the last iteration.
        Composed parts of the RNN from other layers, which are already implemented.
        :param input_tensor: shape = (time, features)
        :return: input tensor for the next layer
        """

        # do the calculation iteratively for every time step
        for time in np.arange(self.bptt_length):

            # check if we are in same sequence and if the hidden states should be reused
            if self.same_sequence is False:
                self.hidden_state = np.zeros(self.hidden_size)
                self.toggle_memory()

            # calculate new hidden state h_t:
            yhh = np.dot(self.hidden_state, self.whh)
            yxh = np.dot(input_tensor[time], self.wxh)
            self.u[time] = yhh + yxh + self.bh
            self.hidden_state = self.list_tanh[time].forward(self.u[time])

            # calculate output y_t:
            sigmoid_input = np.dot(self.hidden_state, self.why) + self.by
            yt = self.list_sigmoid[time].forward(sigmoid_input)

            self.output[time] = yt

        return self.output

    def backward(self, error_tensor):
        """
        Go “backwards” through the unfolded unit, starting at final time step t iteratively compute gradients for
        t = T , ..., 1
        :param error_tensor:
        :return:
        """
        # taken from script, page 18
        # delta_o,t = sigmoid' * dL
        # delta_why,t = delta_o,t * h,t
        # delta_by,t = delta_o,t

        # go through time backward
        for time in np.arange(self.bptt_length)[::-1]:
            # check if this is most future Time step
            if time == (self.bptt_length - 1):
                # calculate delta h_t
                # ToDo acutally self.why should be transposed
                self.delta_h_t = np.dot(self.why, error_tensor[time])
            # do calculation for every normal step between 0 and last
            else:
                hidden_state_influence = np.dot(self.why, error_tensor[time])
                tanh_backward = self.tanh.backward(self.u[time])
