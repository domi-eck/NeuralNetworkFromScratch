import numpy as np
from Layers import Sigmoid
from Layers import TanH
from Layers import FullyConnected
import copy
from Optimization import Optimizers

class LSTM:
    def __init__(self, input_size, hidden_size, output_size, bptt_lenght):
        #region Sizes
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.bptt_length = bptt_lenght
        #endregion Sizes

        self.hidden_state = np.zeros([self.bptt_length, self.hidden_size])
        self.cell_state = np.zeros([self.bptt_length, 4 * self.hidden_size])

        #region Sigmoids
        self.sigmoid_f = []
        self.sigmoid_i = []
        self.sigmoid_o = []

        for i in np.arange(0, self.bptt_length):
            a = Sigmoid.Sigmoid()
            self.sigmoid_f.append(a)
        for i in np.arange(0, self.bptt_length):
            a = Sigmoid.Sigmoid()
            self.sigmoid_i.append(a)
        for i in np.arange(0, self.bptt_length):
            a = Sigmoid.Sigmoid()
            self.sigmoid_o.append(a)
        #endregion

        #region TanH
        self.tanh_o = []
        self.tanh_c = []

        for i in np.arange(0, self.bptt_length):
            c = TanH.TanH()
            self.tanh_o.append(c)
        for i in np.arange(0, self.bptt_length):
            c = TanH.TanH()
            self.tanh_c.append(c)
        #endregion

        #region FullyConnected
        self.list_fully_connected_f  = []
        self.list_fully_connected_i  = []
        self.list_fully_connected_c  = []
        self.list_fully_connected_y  = []
        self.list_fully_connected_o  = []
        self.list_fully_connected_ch = []

        for i in np.arange(0, self.bptt_length):
            a = FullyConnected.FullyConnected(self.hidden_size+self.input_size, 4*self.hidden_size)
            self.list_fully_connected_f.append(a)
        for i in np.arange(0, self.bptt_length):
            a = FullyConnected.FullyConnected(self.hidden_size+self.input_size, 4*self.hidden_size)
            self.list_fully_connected_i.append(a)
        for i in np.arange(0, self.bptt_length):
            a = FullyConnected.FullyConnected(self.hidden_size+self.input_size, 4*self.hidden_size)
            self.list_fully_connected_c.append(a)
        for i in np.arange(0, self.bptt_length):
            a = FullyConnected.FullyConnected(self.hidden_size, output_size)
            self.list_fully_connected_y.append(a)
        for i in np.arange(0, self.bptt_length):
            a = FullyConnected.FullyConnected(self.hidden_size+self.input_size, self.hidden_size)
            self.list_fully_connected_o.append(a)
        for i in np.arange(0, self.bptt_length):
            a = FullyConnected.FullyConnected(4*self.hidden_size, self.hidden_size)
            self.list_fully_connected_ch.append(a)
        #endregion FullyConnected

        #region Metavariables
        self.same_sequence = False
        self.has_optimizer = False
        #endregion

        #region Optimizer
        self.optimizer = None
        #endregion Optimizer


    def toggle_memory(self):
        if self.same_sequence:
            self.same_sequence = False
        else:
            self.same_sequence = True

    def forward(self, input_tensor):
        #concatination
        self.input_tensor = input_tensor
        self.output         = np.zeros([self.bptt_length, self.output_size])

        if self.same_sequence is False:
            self.hidden_state = np.zeros([self.bptt_length , self.hidden_size])
            self.cell_state   = np.zeros([self.bptt_length, 4*self.hidden_size])

        for time in np.arange(input_tensor.shape[0]):
            self.x_tilde = np.hstack([input_tensor[time], self.hidden_state[time]])

            #region Calculating f, i, ctilde, o
            #f
            f       = self.list_fully_connected_f[time].forward(np.expand_dims(self.x_tilde, 0))[0]
            f       = self.sigmoid_f[time].forward(f)
            i       = self.list_fully_connected_i[time].forward(np.expand_dims(self.x_tilde, 0))[0]
            i       = self.sigmoid_i[time].forward(i)
            c_tilde = self.list_fully_connected_c[time].forward(np.expand_dims(self.x_tilde, 0))[0]
            c_tilde = self.tanh_c[time].forward(c_tilde)
            o       = self.list_fully_connected_o[time].forward(np.expand_dims(self.x_tilde, 0))[0]
            o       = self.sigmoid_o[time].forward(o)
            #endregion

            #region Calculating C
            iAndc_tilde                                     = np.multiply(i, c_tilde)
            fAndc                                           = np.multiply(self.cell_state[time], f)
            self.cell_state[(time + 1) % self.bptt_length]  = fAndc + iAndc_tilde
            #endregion

            #region Updating HiddenState
            tanhO                                               = self.list_fully_connected_ch[time].forward(np.expand_dims(self.cell_state[(time + 1) % self.bptt_length], 0))[0]
            tanhO                                               = self.tanh_o[time].forward(tanhO)
            self.hidden_state[(time + 1) % self.bptt_length]    = np.multiply(tanhO, o)
            #endregion

            #region Calculating Output
            y = self.list_fully_connected_y[time].forward(np.expand_dims( self.hidden_state[(time + 1) % self.bptt_length], 0 ))

            self.output[time] = y
            #endregion

        return self.output


    def backward(self, error_tensor):
        dummy = 1

    def get_gradient_weights(self):
        dummy = 1

    def get_weights(self):
        dummy = 1

    def set_weights(self, weights):
        dummy = 1

    def set_optimizer(self, optimizer):
        self.optimizer      = optimizer
        self.has_optimizer  = True

    def get_Loss(self):
        dummy = 1

    def initialize(self, weights_initializer, bias_initializer):
        for iteration in np.arange(self.bptt_length):
            if iteration == 0:
                self.list_fully_connected_f[iteration].initialize(weights_initializer, bias_initializer)
                self.list_fully_connected_i[iteration].initialize(weights_initializer, bias_initializer)
                self.list_fully_connected_c[iteration].initialize(weights_initializer, bias_initializer)
                self.list_fully_connected_o[iteration].initialize(weights_initializer, bias_initializer)
                self.list_fully_connected_y[iteration].initialize(weights_initializer, bias_initializer)
                self.list_fully_connected_ch[iteration].initialize(weights_initializer, bias_initializer)

                self.f_weights  = self.list_fully_connected_f[iteration].get_weights()
                self.f_bias     = self.list_fully_connected_f[iteration].bias
                self.i_weights  = self.list_fully_connected_i[iteration].get_weights()
                self.i_bias     = self.list_fully_connected_i[iteration].bias
                self.c_weights  = self.list_fully_connected_c[iteration].get_weights()
                self.c_bias     = self.list_fully_connected_c[iteration].bias
                self.y_weights  = self.list_fully_connected_y[iteration].get_weights()
                self.y_bias     = self.list_fully_connected_y[iteration].bias
                self.o_weights  = self.list_fully_connected_o[iteration].get_weights()
                self.o_bias     = self.list_fully_connected_o[iteration].bias
                self.ch_weights  = self.list_fully_connected_ch[iteration].get_weights()
                self.ch_bias     = self.list_fully_connected_ch[iteration].bias
            else:
                self.list_fully_connected_f[iteration].set_weights(self.f_weights)
                self.list_fully_connected_f[iteration].set_bias(self.f_bias)
                self.list_fully_connected_i[iteration].set_weights(self.i_weights)
                self.list_fully_connected_i[iteration].set_bias(self.i_bias)
                self.list_fully_connected_c[iteration].set_weights(self.c_weights)
                self.list_fully_connected_c[iteration].set_bias(self.c_bias)
                self.list_fully_connected_y[iteration].set_weights(self.y_weights)
                self.list_fully_connected_y[iteration].set_bias(self.y_bias)
                self.list_fully_connected_o[iteration].set_weights(self.o_weights)
                self.list_fully_connected_o[iteration].set_bias(self.o_bias)
                self.list_fully_connected_ch[iteration].set_weights(self.ch_weights)
                self.list_fully_connected_ch[iteration].set_bias(self.ch_bias)

