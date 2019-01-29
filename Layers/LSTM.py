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

        self.error_ht1      = np.zeros([self.bptt_length, self.hidden_size])
        self.error_cell     = np.zeros([self.bptt_length, self.hidden_size])

        self.learning_rate = 0.001
        self.hidden_state = np.zeros([self.bptt_length, self.hidden_size])
        self.cell_state   = np.zeros([self.bptt_length, self.hidden_size])

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
        self.list_fully_connected_y  = []
        self.list_fully_connected_fico = []

        for i in np.arange(0, self.bptt_length):
            a = FullyConnected.FullyConnected(self.hidden_size+self.input_size, 4*self.hidden_size)
            self.list_fully_connected_fico.append(a)

        for i in np.arange(0, self.bptt_length):
            a = FullyConnected.FullyConnected(self.hidden_size, output_size)
            self.list_fully_connected_y.append(a)

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
        self.f              = np.zeros([self.bptt_length, self.hidden_size])
        self.i              = np.zeros([self.bptt_length, self.hidden_size])
        self.c_tilde        = np.zeros([self.bptt_length, self.hidden_size])
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
            self.cell_state   = np.zeros([self.bptt_length,  self.hidden_size])

        for time in np.arange(input_tensor.shape[0]):
            self.x_tilde = np.hstack([input_tensor[time], self.hidden_state[time]])

            fico = self.list_fully_connected_fico[time].forward(np.expand_dims(self.x_tilde, 0))[0]
            #region Calculating f, i, ctilde, o
            #f
            f                   = fico[0:self.hidden_size]
            self.f[time]        = self.sigmoid_f[time].forward(f)

            i                   = fico[self.hidden_size:self.hidden_size*2]
            self.i[time]        = self.sigmoid_i[time].forward(i)

            c_tilde             = fico[self.hidden_size*2:self.hidden_size*3]
            self.c_tilde[time]  = self.tanh_c[time].forward(c_tilde)

            o                   = fico[self.hidden_size*3:]
            self.o[time]        = self.sigmoid_o[time].forward(o)
            #endregion

            #region Calculating C
            iAndc_tilde                         = np.multiply(self.i[time], self.c_tilde[time])
            fAndc                               = np.multiply(self.cell_state[time], self.f[time])
            self.cell_state[(time + 1) % self.bptt_length]  = fAndc + iAndc_tilde
            #endregion

            #region Updating HiddenState
            self.tanhO[time]                                    = self.tanh_o[time].forward(self.cell_state[(time + 1) % self.bptt_length])
            self.hidden_state[(time + 1) % self.bptt_length]    = np.multiply(self.tanhO[time], self.o[time])
            #endregion

            #region Calculating Output
            y = self.list_fully_connected_y[time].forward(np.expand_dims( self.hidden_state[(time + 1) % self.bptt_length], 0 ))

            self.output[time] = y
            #endregion

        return self.output

    def backward(self, error_tensor):
        self.sum_Gradients_y    = np.zeros_like(self.list_fully_connected_y[-1].get_weights())
        self.sum_Gradients_fico = np.zeros_like(self.list_fully_connected_fico[-1].get_weights())

        self.error_tensor   = error_tensor

        self.output_x       = np.zeros([self.bptt_length, self.input_size])

        if self.same_sequence is False:
            self.error_ht1 = np.zeros([self.bptt_length ,self.hidden_size])
            self.error_cell = np.zeros([self.bptt_length, self.hidden_size])

        for time in np.arange(error_tensor.shape[0])[::-1]:
            error = self.error_tensor[time]

            dh = self.list_fully_connected_y[time].backward(np.expand_dims(error,0))[0]
            if time != self.bptt_length-1:
                dh = dh + self.error_ht1[time+1]

            bo = np.multiply( self.tanhO[time], dh )
            bo = self.sigmoid_o[time].backward(bo)

            dc = np.multiply( self.o[time], dh )
            dc = self.tanh_o[time].backward(dc)
            if time != self.bptt_length-1:
                dc = dc + self.error_cell[time+1]
            self.error_cell[time] = np.multiply( self.f[time], dc )

            if(time != 0):
                bf = self.cell_state[time] * dc
            else:
                bf = np.zeros([self.hidden_size])
            bf = self.sigmoid_f[time].backward(bf)

            bi          = np.multiply( self.c_tilde[time], dc )
            bi          = self.sigmoid_i[time].backward(bi)

            bc_tilde    = np.multiply(self.i[time], dc)
            bc_tilde    = self.tanh_c[time].backward(bc_tilde)

            xh = np.hstack([ bf, bi, bc_tilde, bo ])
            xh = self.list_fully_connected_fico[time].backward(np.expand_dims(xh, 0))[0]


            #Get gradients
            self.sum_Gradients_y = self.sum_Gradients_y + self.list_fully_connected_y[time].get_gradient_weights()
            self.sum_Gradients_fico = self.sum_Gradients_fico + self.list_fully_connected_fico[time].get_gradient_weights()


            self.error_ht1[time] = xh[self.input_size:]
            self.output_x[time] = xh[:self.input_size]

        #region Update
        # optimize the weights
        if self.has_optimizer is True:
            self.y_weights = self.optimizer.calculate_update(self.learning_rate, self.y_weights, self.sum_Gradients_y)
            self.fico_weights = self.optimizer.calculate_update(self.learning_rate, self.fico_weights, self.sum_Gradients_fico)

            for layer in self.list_fully_connected_fico:
                layer.set_weights(self.fico_weights)
            for layer in self.list_fully_connected_y:
                layer.set_weights(self.y_weights)

        #endregion

        return  self.output_x



    def get_gradient_weights(self):
        return self.sum_Gradients_fico

    def get_weights(self):
        return self.fico_weights

    def set_weights(self, weights):
        self.fico_weights = weights
        for iteration in np.arange(self.bptt_length):
            self.list_fully_connected_fico[iteration].set_weights(self.fico_weights)

    def set_optimizer(self, optimizer):
        self.optimizer      = optimizer
        self.has_optimizer  = True

    def get_Loss(self):
        dummy = 1

    def initialize(self, weights_initializer, bias_initializer):
        for iteration in np.arange(self.bptt_length):
            if iteration == 0:
                self.list_fully_connected_y[iteration].initialize(weights_initializer, bias_initializer)
                self.list_fully_connected_fico[iteration].initialize(weights_initializer, bias_initializer)

                self.y_weights  = self.list_fully_connected_y[iteration].get_weights()
                self.y_bias = self.list_fully_connected_y[iteration].bias
                self.fico_weights = self.list_fully_connected_fico[iteration].get_weights()
                self.fico_bias = self.list_fully_connected_fico[iteration].bias
            else:
                self.list_fully_connected_y[iteration].set_weights(self.y_weights)
                self.list_fully_connected_y[iteration].set_bias(self.y_bias)
                self.list_fully_connected_fico[iteration].set_weights(self.fico_weights)
                self.list_fully_connected_fico[iteration].set_bias(self.fico_bias)


