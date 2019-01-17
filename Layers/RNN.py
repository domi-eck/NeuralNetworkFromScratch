import numpy as np
from Layers import Sigmoid

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

        # init activation functions
        self.sigmoid = Sigmoid.Sigmoid()

        # init output
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

        # init output
        self.output = np.zeros([self.bptt_length, self.output_size])

        # do the calculation iteratively for every time step
        for time in np.arange(self.bptt_length):

            # check if we are in same sequence and if the hidden states should be reused
            if self.same_sequence is False:
                self.hidden_state = np.zeros(self.hidden_size)
                self.toggle_memory()

            # calculate new hidden state:
            yhh = np.dot(self.hidden_state, self.whh)
            yxh = np.dot(input_tensor[time], self.wxh)
            self.hidden_state = np.tanh(yhh + yxh + self.bh)

            # calculate output
            sigmoid_input = np.dot(self.hidden_state, self.why) + self.by
            yt = self.sigmoid.forward(sigmoid_input)

            self.output[time] = yt

        return self.output

