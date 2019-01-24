import numpy as np
from Layers import Sigmoid
from Layers import TanH
from Layers import FullyConnected
from Optimization import Optimizers


class RNN:
    def __init__(self, input_size, hidden_size, output_size, bptt_length):
        """
        Recurrent Neuronal Network:
            1. ot = ht*Wy
            2. ht = tanh(at)
            3. at = b + h(t-1)*Wh + xt*Wx
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

        self.hidden_state = np.zeros([self.bptt_length, self.hidden_size])
        self.same_sequence = False

        # at = b + h(t-1)*Wh + xt*Wx = b + ah + ax
        self.ax = [np.zeros(self.hidden_size)] * self.bptt_length
        self.ah = [np.zeros(self.hidden_size)] * self.bptt_length
        self.a = [np.zeros(self.hidden_size)] * self.bptt_length

        # parameters for backward
        self.hidden_gradients = np.zeros([self.bptt_length + 1, self.hidden_size])
        self.xt_weight_gradients = np.zeros([self.bptt_length, self.input_size + 1, self.hidden_size])
        self.ht_weight_gradients = np.zeros([self.bptt_length, self.hidden_size + 1, self.hidden_size])
        self.yt_weight_gradients = np.zeros([self.bptt_length, self.hidden_size + 1, self.output_size])

        # error which should be past out of the whole RNN
        self.error_xt = np.zeros([self.bptt_length, self.input_size])

        # init the optimizer parameter
        self.has_optimizer = False
        self.optimizer = None
        self.learning_rate = 1

        # Activations and weights of every time step are needed for the backward pass
        # therefore the activations are stored in the objects of Sigmoid and TanH class,
        # there is a instance for every time step: """
        # init Tanh functions, also to store them for the backward algorithm
        self.list_tanh = [TanH.TanH()] * self.bptt_length


        # init output with zeros
        self.output = np.zeros([self.bptt_length, self.output_size])

        # fully connected instances; concatenated input -> [h_(t-1), x_t, b]
        self.list_fully_connected_xt = \
            [FullyConnected.FullyConnected(self.input_size, self.hidden_size)] * self.bptt_length
        self.list_fully_connected_ht = \
            [FullyConnected.FullyConnected(self.hidden_size, self.hidden_size)] * self.bptt_length
        self.list_fully_connected_yt = \
            [FullyConnected.FullyConnected(self.hidden_size, self.output_size)] * self.bptt_length


        # initialize the weights
        self.xt_weights = np.random.rand(self.input_size + 1, self.hidden_size)
        self.ht_weights = np.random.rand(self.hidden_size + 1, self.hidden_size)
        self.yt_weights = np.random.rand(self.hidden_size + 1, self.output_size)

        for layer in self.list_fully_connected_xt:
            layer.set_weights(self.xt_weights)
            # init bias to zero,
            layer.bias = np.zeros(self.hidden_size)

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
        Reminder and Notation, forward pass
        3. ot = ht*Wy
        2. ht = tanh(at)
        1. at = b + h(t-1)*Wh + xt*Wx

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

            # calculate at
            self.ax[time] = self.list_fully_connected_xt[time].forward(np.expand_dims(input_tensor[time], 0))[0]
            self.ah[time] = self.list_fully_connected_ht[time].forward(np.expand_dims(self.hidden_state[time], 0))[0]
            self.a[time] = self.ax[time] + self.ah[time]

            # modulo operation two write last hidden state back to the first hidden state, so it can be used
            # by the next forward call if same_sequence is True
            self.hidden_state[(time + 1) % self.bptt_length] = self.list_tanh[time].forward(self.a[time])

            # calculate output y_t:
            yt = self.list_fully_connected_yt[time].forward(
                np.expand_dims(self.hidden_state[(time + 1) % self.bptt_length], 0))
            self.output[time] = yt

        return self.output

    def backward(self, error_tensor):
        """
        Go “backwards” through the unfolded unit, starting at final time step t iteratively compute gradients for
        t = T , ..., 1
        :param error_tensor:
        :return:
        """

        for time in np.arange(error_tensor.shape[0])[::-1]:
            # Reminder and Notation, forward pass
            # 1. ot = ht*Wy
            # 2. ht = tanh(at)
            # 3. at = b + h(t-1)*Wh + xt*Wx

            # calculate first ght
            # ght = dL/dht = dot/dht * got
            got = error_tensor[time]
            ght = self.list_fully_connected_yt[time].backward(np.expand_dims(got, 0))[0]

            # calculate first gat
            # dL/dat = dht/dat*ght
            gat = self.list_tanh[time].backward(ght)

            # dL/dxt = dat/dxt * gat
            self.error_xt[time] = self.list_fully_connected_xt[time].backward(np.expand_dims(gat, 0))[0]

            # Calculate the gradients for Wx and Wy for the given timestep
            # dL/dWx = dat/dWx *gat
            self.xt_weight_gradients[time] = self.list_fully_connected_xt[time].get_gradient_weights()

            # with got, dLdWy can be calculated
            self.yt_weight_gradients[time] = self.list_fully_connected_yt[time].get_gradient_weights()

            # init gWh
            # dL/dWh = [dat/dWh + dh(t-1)/dWh * dat/dh(t-1)] * gat
            # dat/dh(t-1) , part of 2.
            ght = self.list_fully_connected_ht[time].backward(np.expand_dims(gat, 0))[0]
            # dat/dWh * gat, 1. on calculation paper
            self.ht_weight_gradients[time] = self.list_fully_connected_ht[time].get_gradient_weights()

            # loop back until bptt_length
            for step in np.arange(max(0, time - self.bptt_length), time)[::-1]:
                # gh(t-2) = da(t-1)/dh(t-2) * gh(t-1)
                ght = self.list_fully_connected_ht[step].backward(np.expand_dims(ght, 0))[0]
                # dh(t-1)/dWh * dat/dh(t-1) * gat = dh(t-1)/dWh * gh(t-1)
                self.ht_weight_gradients[time] += self.list_fully_connected_ht[step].get_gradient_weights()


        # calculate the gradients for update
        sum_ht_gradient = np.sum(self.ht_weight_gradients, 0)
        sum_yt_gradient = np.sum(self.yt_weight_gradients, 0)
        sum_xt_gradient = np.sum(self.xt_weight_gradients, 0)

        self.ht_gradient = sum_ht_gradient
        self.xt_gradient = sum_xt_gradient

        # optimize the weights
        if self.has_optimizer is True:
            self.yt_weights = self.optimizer.calculate_update(self.learning_rate, self.yt_weights, sum_yt_gradient)
            self.ht_weights = self.optimizer.calculate_update(self.learning_rate, self.ht_weights, sum_ht_gradient)
            self.xt_weights = self.optimizer.calculate_update(self.learning_rate, self.xt_weights, sum_xt_gradient)
        else:
            self.ht_weights = self.ht_weights - self.learning_rate*sum_ht_gradient
            self.yt_weights = self.yt_weights - self.learning_rate*sum_yt_gradient
            self.xt_weights = self.xt_weights - self.learning_rate * sum_xt_gradient

        for layer in self.list_fully_connected_ht:
            layer.set_weights(self.ht_weights)
        for layer in self.list_fully_connected_yt:
            layer.set_weights(self.yt_weights)
        for layer in self.list_fully_connected_xt:
            layer.set_weights(self.xt_weights)

        return self.error_xt

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.has_optimizer = True

    def initialize(self, weights_initializer, bias_initializer):
        for iteration in np.arange(self.bptt_length):
            if iteration == 0:
                self.list_fully_connected_yt[iteration].initialize(weights_initializer, bias_initializer)
            else:
                self.list_fully_connected_yt[iteration].set_weights(self.list_fully_connected_yt[0].get_weights())
                self.list_fully_connected_yt[iteration].set_bias(self.list_fully_connected_yt[0].bias)

        for iteration in np.arange(self.bptt_length):
            if iteration == 0:
                self.list_fully_connected_ht[iteration].initialize(weights_initializer, bias_initializer)
            else:
                self.list_fully_connected_ht[iteration].set_weights(self.list_fully_connected_ht[0].get_weights())
                self.list_fully_connected_ht[iteration].set_bias(self.list_fully_connected_ht[0].bias)

        for iteration in np.arange(self.bptt_length):
            if iteration == 0:
                self.list_fully_connected_xt[iteration].initialize(weights_initializer, bias_initializer)
                self.list_fully_connected_xt[iteration].bias = np.zeros(self.hidden_size)
            else:
                self.list_fully_connected_xt[iteration].set_weights(self.list_fully_connected_xt[0].get_weights())
                self.list_fully_connected_xt[iteration].set_bias(self.list_fully_connected_xt[0].bias)

    def get_weights(self):
        return self.ht_weights

    def set_weights(self, weights):
        self.ht_weights = weights

    def get_gradient_weights(self):
        return self.ht_gradient



