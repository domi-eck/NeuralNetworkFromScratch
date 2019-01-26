import numpy as np
from Layers import Sigmoid
from Layers import TanH
from Layers import FullyConnected
import copy
from Optimization import Optimizers


class RNN:
    def __init__(self, input_size, hidden_size, output_size, bptt_length):
        self.input_size     = input_size
        self.hidden_size    = hidden_size
        self.output_size    = output_size
        self.bptt_length    = bptt_length  # (batch size bzw. time dimension)

        self.hidden_state = np.zeros([self.bptt_length, self.hidden_size])
        self.same_sequence = False

        #TODO Probabily not needed
        # at = b + h(t-1)*Wh + xt*Wx = b + ah + ax
        self.ax = [np.zeros(self.hidden_size)] * self.bptt_length
        self.ah = [np.zeros(self.hidden_size)] * self.bptt_length
        self.a = [np.zeros(self.hidden_size)] * self.bptt_length

        # parameters for backward
        self.hidden_gradients = np.zeros([self.bptt_length + 1, self.hidden_size])


        # error which should be past out of the whole RNN
        self.error_xt = np.zeros([self.bptt_length, self.input_size])
        # error from h
        self.error_ht = np.zeros([self.bptt_length, self.hidden_size])

        # init the optimizer parameter
        self.has_optimizer = False
        self.optimizer = None
        self.learning_rate = 1

        # init output with zeros
        self.output = np.zeros([self.bptt_length, self.output_size])

        self.list_fully_connected_xhh   = []
        self.list_fully_connected_hy    = []
        self.list_tanh                  = []

        for i in np.arange(0,self.bptt_length):
            a = FullyConnected.FullyConnected(self.hidden_size+self.input_size, self.hidden_size)
            self.list_fully_connected_xhh.append(a)
        for i in np.arange(0,self.bptt_length):
            b = FullyConnected.FullyConnected(self.hidden_size, self.output_size)
            self.list_fully_connected_hy.append(b)
        for i in np.arange(0,self.bptt_length):
            c = TanH.TanH()
            self.list_tanh.append(c)

        # initialize the weights

        self.hy_weight_gradients = np.zeros([self.bptt_length, self.hidden_size + 1, self.output_size])
        self.xhh_weight_gradients = np.zeros([self.bptt_length, self.hidden_size + 1 + self.input_size, self.hidden_size])

        self.hy_weights = np.random.rand(self.hidden_size + 1, self.output_size)
        self.xhh_weights = np.random.rand(self.hidden_size + 1 + self.input_size, self.hidden_size)

        for layer in self.list_fully_connected_xhh:
            layer.set_weights(self.xhh_weights)
        for layer in self.list_fully_connected_hy:
            layer.set_weights(self.hy_weights)

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
        # ToDo why is this init needed here?
        self.output = np.zeros([self.bptt_length, self.output_size])

        if self.same_sequence is False:
            self.hidden_state = np.zeros([self.bptt_length ,self.hidden_size])

        for time in np.arange(input_tensor.shape[0]):

            # check if we are in same sequence and if the hidden states should be reused


            #stacking input and previos h #TODO check where x and h is in xTilde
            xTilde = np.hstack([input_tensor[time], self.hidden_state[time]])
            self.a[time] = self.list_fully_connected_xhh[time].forward(np.expand_dims(xTilde, 0))[0]

            # modulo operation two write last hidden state back to the first hidden state, so it can be used
            # by the next forward call if same_sequence is True
            self.hidden_state[(time + 1) % self.bptt_length] = self.list_tanh[time].forward(self.a[time])

            # calculate output y_t:
            yt = self.list_fully_connected_hy[time].forward(
                np.expand_dims(self.hidden_state[(time + 1) % self.bptt_length], 0))
            self.output[time] = yt

        return self.output

    def backward(self, error_tensor):

        if self.same_sequence is False:
            self.error_ht = np.zeros([self.bptt_length ,self.hidden_size])

        for time in np.arange(error_tensor.shape[0])[::-1]:
            dy = error_tensor[time]
            dh = self.list_fully_connected_hy[time].backward(np.expand_dims(dy, 0))[0]
            if( time != self.bptt_length-1):
                dh = dh + self.error_ht[time+1]

            gat = self.list_tanh[time].backward(dh)

            dxTilde = self.list_fully_connected_xhh[time].backward(np.expand_dims(gat, 0))[0]

            self.error_xt[time] = dxTilde[0:self.input_size]
            self.error_ht[time] = dxTilde[self.input_size:]

            #getting the weights gradients for updates
            #dL/dW_hy
            self.hy_weight_gradients[time] = self.list_fully_connected_hy[time].get_gradient_weights()
            #dL/dW_xhh
            self.xhh_weight_gradients[time] = self.list_fully_connected_xhh[time].get_gradient_weights()


        # calculate the gradients for update
        self.sum_hy_weight_gradient = np.sum(self.hy_weight_gradients, 0)
        self.sum_xhh_weight_gradient = np.sum(self.xhh_weight_gradients, 0)

        # optimize the weights
        if self.has_optimizer is True:
            self.hy_weights = self.optimizer.calculate_update(self.learning_rate, self.hy_weights, self.sum_hy_weight_gradient)
            self.xhh_weights = self.optimizer.calculate_update(self.learning_rate, self.xhh_weights, self.sum_xhh_weight_gradient)

            for layer in self.list_fully_connected_hy:
                layer.set_weights(self.hy_weights)
            for layer in self.list_fully_connected_xhh:
                layer.set_weights(self.xhh_weights)

        return self.error_xt

    #checked
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self.has_optimizer = True

    #checked
    def initialize(self, weights_initializer, bias_initializer):
        for iteration in np.arange(self.bptt_length):
            if iteration == 0:
                self.list_fully_connected_hy[iteration].initialize(weights_initializer, bias_initializer)
                self.hy_weights = self.list_fully_connected_hy[iteration].get_weights()
                self.by_bias = self.list_fully_connected_hy[iteration].bias
            else:
                self.list_fully_connected_hy[iteration].set_weights(self.hy_weights)
                self.list_fully_connected_hy[iteration].set_bias(self.by_bias)
        for iteration in np.arange(self.bptt_length):
            if iteration == 0:
                self.list_fully_connected_xhh[iteration].initialize(weights_initializer, bias_initializer)
                self.xhh_weights = self.list_fully_connected_xhh[iteration].get_weights()
                self.xhh_bias = self.list_fully_connected_xhh[iteration].bias
            else:
                self.list_fully_connected_xhh[iteration].set_weights(self.xhh_weights)
                self.list_fully_connected_xhh[iteration].set_bias(self.xhh_bias)


    def get_weights(self):
        return self.hy_weights

    def set_weights(self, weights):
        self.hy_weights = weights
        for iteration in np.arange(self.bptt_length):
            self.list_fully_connected_hy[iteration].set_weights(self.hy_weights)

    def get_gradient_weights(self):
        return self.sum_hy_weight_gradient



