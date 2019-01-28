import numpy as np
from Layers import Sigmoid
from Layers import TanH
from Layers import FullyConnected
from Layers import Initializers
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

        self.learning_rate = 0.001
        self.hidden_state = np.zeros([self.bptt_length, self.hidden_size])
        self.cell_state   = np.zeros([self.bptt_length, 4 * self.hidden_size])

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

        self.initialize(Initializers.He(), Initializers.He())
        #endregion FullyConnected

        #region Metavariables
        self.same_sequence = False
        self.has_optimizer = False
        #endregion

        #region Optimizer
        self.optimizer = None
        #endregion Optimizer

        #region Input for Multiplication
        self.f              = np.zeros([self.bptt_length, 4*self.hidden_size])
        self.i              = np.zeros([self.bptt_length, 4*self.hidden_size])
        self.c_tilde        = np.zeros([self.bptt_length, 4*self.hidden_size])
        self.tanhO          = np.zeros([self.bptt_length, self.hidden_size])
        self.o              = np.zeros([self.bptt_length, self.hidden_size])
        #endregion

    def toggle_memory(self):
        if self.same_sequence:
            self.same_sequence = False
        else:
            self.same_sequence = True

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        self.output       = np.zeros([self.bptt_length, self.output_size])

        if self.same_sequence is False:
            self.hidden_state = np.zeros([self.bptt_length , self.hidden_size])
            self.cell_state   = np.zeros([self.bptt_length, 4*self.hidden_size])

        for time in np.arange(input_tensor.shape[0]):
            self.x_tilde = np.hstack([input_tensor[time], self.hidden_state[time]])

            #region Calculating f, i, ctilde, o
            #f
            f                   = self.list_fully_connected_f[time].forward(np.expand_dims(self.x_tilde, 0))[0]
            self.f[time]        = self.sigmoid_f[time].forward(f)
            i                   = self.list_fully_connected_i[time].forward(np.expand_dims(self.x_tilde, 0))[0]
            self.i[time]        = self.sigmoid_i[time].forward(i)
            c_tilde             = self.list_fully_connected_c[time].forward(np.expand_dims(self.x_tilde, 0))[0]
            self.c_tilde[time]  = self.tanh_c[time].forward(c_tilde)
            o                   = self.list_fully_connected_o[time].forward(np.expand_dims(self.x_tilde, 0))[0]
            self.o[time]        = self.sigmoid_o[time].forward(o)
            #endregion

            #region Calculating C
            iAndc_tilde                         = np.multiply(self.i[time], self.c_tilde[time])
            fAndc                              = np.multiply(self.cell_state[time], self.f[time])
            self.cell_state[(time + 1) % self.bptt_length]  = fAndc + iAndc_tilde
            #endregion

            #region Updating HiddenState
            self.tanhO[time]                                    = self.list_fully_connected_ch[time].forward(np.expand_dims(self.cell_state[(time + 1) % self.bptt_length], 0))[0]
            self.tanhO[time]                                    = self.tanh_o[time].forward(self.tanhO[time])
            self.hidden_state[(time + 1) % self.bptt_length]    = np.multiply(self.tanhO[time], o)
            #endregion

            #region Calculating Output
            y = self.list_fully_connected_y[time].forward(np.expand_dims( self.hidden_state[(time + 1) % self.bptt_length], 0 ))

            self.output[time] = y
            #endregion

        return self.output

    def backward(self, error_tensor):
        self.sum_Gradients_f = np.zeros_like(self.list_fully_connected_f[-1].get_weights())
        self.sum_Gradients_i = np.zeros_like(self.list_fully_connected_i[-1].get_weights())
        self.sum_Gradients_c = np.zeros_like(self.list_fully_connected_c[-1].get_weights())
        self.sum_Gradients_o = np.zeros_like(self.list_fully_connected_o[-1].get_weights())
        self.sum_Gradients_y = np.zeros_like(self.list_fully_connected_y[-1].get_weights())
        self.sum_Gradients_ch = np.zeros_like(self.list_fully_connected_ch[-1].get_weights())

        self.error_tensor   = error_tensor
        self.error_ht1      = np.zeros([self.bptt_length, self.hidden_size])
        self.error_cell     = np.zeros([self.bptt_length, 4*self.hidden_size])
        self.output_x       = np.zeros([self.bptt_length, self.input_size])

        if self.same_sequence is False:
            self.error_ht1 = np.zeros([self.bptt_length ,self.hidden_size])
            self.error_cell = np.zeros([self.bptt_length, 4*self.hidden_size])

        for time in np.arange(error_tensor.shape[0])[::-1]:
            error = self.error_tensor[time]

            dh = self.list_fully_connected_y[time].backward(np.expand_dims(error,0))
            if time != self.bptt_length-1:
                dh = dh + self.error_ht1[time+1]

            bo = np.multiply( self.tanhO[time], dh )
            bo = self.sigmoid_o[time].backward(bo)
            bo = self.list_fully_connected_o[time].backward(bo)

            dc = np.multiply( self.o[time], dh )
            dc = self.tanh_o[time].backward(dc)
            dc = self.list_fully_connected_ch[time].backward(dc)
            if time != self.bptt_length-1:
                dc = dc + self.error_cell[time+1]

            self.error_cell[time] = np.multiply( self.f[time], dc )
            bf = self.cell_state[time] * dc
            bf = self.sigmoid_f[time].backward(bf)
            bf = self.list_fully_connected_f[time].backward(bf)

            bi          = np.multiply( self.c_tilde[time], dc )
            bi          = self.sigmoid_i[time].backward(bi)
            bi          = self.list_fully_connected_i[time].backward(bi)

            bc_tilde    = np.multiply(self.i[time], dc)
            bc_tilde    = self.tanh_c[time].backward(bc_tilde)
            bc_tilde    = self.list_fully_connected_c[time].backward(bc_tilde)

            xh = bc_tilde + bi + bf + bo

            #Get gradients
            self.sum_Gradients_f = self.sum_Gradients_f + self.list_fully_connected_f[time].get_gradient_weights()
            self.sum_Gradients_i = self.sum_Gradients_i + self.list_fully_connected_i[time].get_gradient_weights()
            self.sum_Gradients_c = self.sum_Gradients_c + self.list_fully_connected_c[time].get_gradient_weights()
            self.sum_Gradients_o = self.sum_Gradients_o + self.list_fully_connected_o[time].get_gradient_weights()
            self.sum_Gradients_y = self.sum_Gradients_y + self.list_fully_connected_y[time].get_gradient_weights()
            self.sum_Gradients_ch = self.sum_Gradients_ch + self.list_fully_connected_ch[time].get_gradient_weights()


           # self.error_ht[time] = xh[0][self.input_size:]
            self.output_x[time] = xh[0][:self.input_size]

        #region Update
        # optimize the weights
        if self.has_optimizer is True:
            self.f_weights = self.optimizer.calculate_update(self.learning_rate, self.f_weights, self.sum_Gradients_f)
            self.i_weights = self.optimizer.calculate_update(self.learning_rate, self.i_weights, self.sum_Gradients_i)
            self.c_weights = self.optimizer.calculate_update(self.learning_rate, self.c_weights, self.sum_Gradients_c)
            self.o_weights = self.optimizer.calculate_update(self.learning_rate, self.o_weights, self.sum_Gradients_o)
            self.y_weights = self.optimizer.calculate_update(self.learning_rate, self.y_weights, self.sum_Gradients_y)
            self.ch_weights = self.optimizer.calculate_update(self.learning_rate, self.ch_weights, self.sum_Gradients_ch)

            for layer in self.list_fully_connected_f:
                layer.set_weights(self.f_weights)
            for layer in self.list_fully_connected_i:
                layer.set_weights(self.i_weights)
            for layer in self.list_fully_connected_c:
                layer.set_weights(self.c_weights)
            for layer in self.list_fully_connected_o:
                layer.set_weights(self.o_weights)
            for layer in self.list_fully_connected_y:
                layer.set_weights(self.y_weights)
            for layer in self.list_fully_connected_ch:
                layer.set_weights(self.ch_weights)

        #endregion

        return  self.output_x



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

