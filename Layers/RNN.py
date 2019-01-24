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

        self.hidden_state = np.zeros([self.bptt_length, self.hidden_size])
        self.same_sequence = False

        # parameters for backward
        self.hidden_gradients = np.zeros([self.bptt_length +1, self.hidden_size])
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
            self.u[time] = self.list_fully_connected_ht[time].forward(np.expand_dims(x_tilde, 0))[0]
            # modulo operation two write last hidden state back to the first hidden state, so it can be used
            # by the next forward call if same_sequence is True
            self.hidden_state[(time + 1) % self.bptt_length] = self.list_tanh[time].forward(self.u[time])

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
            # Initial delta calculation: dL/dz
            # self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
            delta_y = self.list_fully_connected_yt[time].backward(np.expand_dims(error_tensor[time], 0))[0]
            da = self.list_tanh[time].backward(delta_y)

            # dLdV += np.outer(delta_o[t], s[t].T)
            self.yt_weight_gradients[time] = self.list_fully_connected_yt[time].get_gradient_weights()
            # go back util truncation_length or 0 in time is reached
            for step in np.arange(max(0, time-self.bptt_length), time + 1)[::-1]:
                delta_hxb = self.list_fully_connected_ht[step].backward(np.expand_dims(delta_h, 0))[0]
                # ToDo: definitely not sure if this is right, could also be last or the sum over all errors
                if step == time:
                    self.error_xt[step] = delta_hxb[self.hidden_size: self.hidden_size + self.input_size]
                delta_h = delta_hxb[0: self.hidden_size]
                if step != 0:
                    delta_h = self.list_tanh[step-1].backward(delta_h)

                self.ht_weight_gradients[time] += self.list_fully_connected_ht[step].get_gradient_weights()

            #self.error_xt[step] = delta_hxb[self.hidden_size: self.hidden_size + self.input_size]



        #     # 2. (dh(t+1)/dht) * gradient_h(t+1) with tanH(Whn*h(t-1) + Wxh*xt + bh)
        #     if time + 1 == self.bptt_length:
        #         delta_h = np.zeros_like(self.hidden_gradients[time])
        #     else:
        #         delta_h = self.list_tanh[time].backward(self.hidden_gradients[time + 1])[0]
        #     # because tis fully connected combines the normal input and the old hidden state, only the output
        #     # for the hidden state is important to the gradient
        #
        #     delta_h = delta_hxb[0: self.hidden_size]
        #
        #     # add this two elements together to the hidden gradient
        #     self.hidden_gradients[time] = delta_y + delta_h
        #
        #     # get the gradients
        #     self.ht_weight_gradients[time] = self.list_fully_connected_ht[time].get_gradient_weights()
        #     self.yt_weight_gradients[time] = self.list_fully_connected_yt[time].get_gradient_weights()
        #
        #     # write the error, which is part of the delta_hxb
        #     self.error_xt[time] = delta_hxb[self.hidden_size: self.input_size + self.hidden_size]
        #
        # calculate the gradients for update
        sum_ht_gradient = np.sum(self.ht_weight_gradients, 0)
        sum_yt_gradient = np.sum(self.yt_weight_gradients, 0)

        self.ht_gradient = sum_ht_gradient

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

    def get_weights(self):
        return self.ht_weights

    def set_weights(self, weights):
        self.ht_weights = weights

    def get_gradient_weights(self):
        return self.ht_gradient



