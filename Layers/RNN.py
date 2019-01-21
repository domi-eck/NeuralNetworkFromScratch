import numpy as np
from Layers import Sigmoid
from Layers import TanH
from Layers import FullyConnected
from Optimization import Optimizers


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
        # parameter for Forward
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bptt_length = bptt_length  # (batch size bzw. time dimension)

        self.hidden_state = np.zeros([self.bptt_length + 1, self.hidden_size])
        self.same_sequence = False

        # parameters for backward
        self.hidden_gradients = np.zeros([self.bptt_length + 1, self.hidden_size])
        self.ht_weight_gradients = np.zeros([self.bptt_length, self.hidden_size + self.input_size + 1, self.hidden_size])
        self.yt_weight_gradients = np.zeros([self.bptt_length, self.hidden_size + 1, self.output_size])

        # error which should be past out of the whole RNN
        self.error_xt = np.zeros([self.bptt_length, self.input_size])

        # init the optimizer parameter
        self.has_optimizer = False
        self.optimizer = None
        self.learning_rate = 1

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

        # fully connected instances; concatenated input -> [h_(t-1), x_t, b]
        self.list_fully_connected_ht = [FullyConnected.FullyConnected(
            (self.hidden_size + self.input_size), self.hidden_size)] * self.bptt_length
        self.list_fully_connected_yt = [FullyConnected.FullyConnected(
            self.hidden_size, self.output_size)] * self.bptt_length

        # initialize the weights
        self.ht_weights = np.random.rand(self.hidden_size + self.input_size + 1, self.hidden_size)
        self.yt_weights = np.random.rand(self.hidden_size + 1, self.output_size)

        for layer in self.list_fully_connected_ht:
            layer.set_weights(self.ht_weights)

        for layer in self.list_fully_connected_yt:
            layer.set_weights(self.yt_weights)

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

        # init output with zeros
        # ToDo why is this init needed here?
        self.output = np.zeros([self.bptt_length, self.output_size])

        # do the calculation iteratively for every time step
        for time in np.arange(input_tensor.shape[0]):

            # check if we are in same sequence and if the hidden states should be reused
            if self.same_sequence is False:
                self.hidden_state[time] = np.zeros(self.hidden_size)
                self.toggle_memory()

            # calculate h_t
            x_tilde = np.concatenate([self.hidden_state[time], input_tensor[time]])
            self.u[time] = self.list_fully_connected_ht[time].forward(np.expand_dims(x_tilde, 0))
            # modulo operation two write last hidden state back to the first hidden state, so it can be used
            # by the next forward call if same_sequence is True
            self.hidden_state[(time + 1) % (self.bptt_length - 1)] = self.list_tanh[time].forward(self.u[time])[0]

            # calculate output y_t:
            yt = self.list_fully_connected_yt[time].forward(
                np.expand_dims(self.hidden_state[(time + 1) % (self.bptt_length - 1)], 0))
            self.output[time] = yt

        return self.output

    def backward(self, error_tensor):
        """
        Go “backwards” through the unfolded unit, starting at final time step t iteratively compute gradients for
        t = T , ..., 1
        :param error_tensor:
        :return:
        """
        # go through time backward
        for time in np.arange(self.bptt_length)[::-1]:

            # calculate delta h_t with consists of two elements:
            # 1. (dot/dht)*gradient_ot with ot = Why*ht + by
            delta_y = self.list_fully_connected_yt[time].backward(np.expand_dims(error_tensor[time], 0))[0]

            # 2. (dh(t+1)/dht) * gradient_h(t+1) with tanH(Whn*h(t-1) + Wxh*xt + bh)
            delta_h = self.list_tanh[time].backward(self.hidden_gradients[time + 1])[0]
            # because tis fully connected combines the normal input and the old hidden state, only the output
            # for the hidden state is important to the gradient
            delta_hxb = self.list_fully_connected_ht[time].backward(np.expand_dims(delta_h, 0))[0]
            delta_h = delta_hxb[0: self.hidden_size]

            # add this two elements together to the hidden gradient
            self.hidden_gradients[time] = delta_y + delta_h

            # get the gradients
            self.ht_weight_gradients[time] = self.list_fully_connected_ht[time].get_gradient_weights()
            self.yt_weight_gradients[time] = self.list_fully_connected_yt[time].get_gradient_weights()

            # write the error, which is part of the delta_hxb
            self.error_xt[time] = delta_hxb[self.hidden_size: self.input_size + self.hidden_size]

        # calculate the gradients for update
        sum_ht_gradient = np.sum(self.ht_weight_gradients, 0)
        sum_yt_gradient = np.sum(self.yt_weight_gradients, 0)

        # optimize the weights
        if self.has_optimizer is True:
            self.yt_weights = self.optimizer.calculate_update(self.learning_rate, self.yt_weights, sum_yt_gradient)
            self.ht_weights = self.optimizer.calculate_update(self.learning_rate, self.ht_weights, sum_ht_gradient)
        else:
            self.ht_weights = self.ht_weights - self.learning_rate*sum_ht_gradient
            self.yt_weights = self.yt_weights - self.learning_rate*sum_yt_gradient

        for layer in self.list_fully_connected_ht:
            layer.set_weights(self.ht_weights)
        for layer in self.list_fully_connected_yt:
            layer.set_weights(self.yt_weights)

        return self.error_xt

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.has_optimizer = True

    def initialize(self, weights_initializer, bias_initializer):
        for layer in self.list_fully_connected_yt:
            layer.initialize(weights_initializer, bias_initializer)
        for layer in self.list_fully_connected_ht:
            layer.initialize(weights_initializer, bias_initializer)

    def get_weights(self):
        return None


# ToDo only one fully connected layer is needed

