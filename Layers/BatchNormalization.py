import  numpy as np
import copy
import Layers.Base as Base

class BatchNormalization(Base.Base):
    def __init__(self, channel = 0):
        self.channels = channel
        self.phase              = Base.Phase.train
        self.epsilon            = 0.0000000000000001
        self.hasBiasAndWeights  = False
        self.shapeOfWeights     = []
        self.shapeOfBias     = []
        self.biasInit     = []
        self.weightsInit     = []
        self.needForInit = False

    def set_optimizer(self, optimizer):
        self.optimizer = copy.deepcopy( optimizer )

    def forward(self, input_tensor):

        self.input_tensor = input_tensor

        if self.channels > 0:
            input_tensor        = np.transpose(input_tensor, (0, 3, 2, 1))
            shapeBevorReshaping = input_tensor.shape
            input_tensor        = input_tensor.reshape(-1, self.channels)

        if self.hasBiasAndWeights == False:
            self.hasBiasAndWeights  = True
            self.bias               = np.zeros(input_tensor.shape[1])
            self.shapeOfBias        = self.bias.shape
            self.weights            = np.ones(input_tensor.shape[1])
            self.shapeOfWeights     = self.weights.shape

            if(self.needForInit):
                self.weights    = self.weightsInitializer.initialize( self.weights.shape, input_tensor.shape[0], input_tensor.shape[0])
                self.bias       = self.weightsInitializer.initialize( self.bias.shape, 1, input_tensor.shape[0])



        if self.phase == Base.Phase.train:
            #####

            N, D = input_tensor.shape

            # step1: calculate mean
            self.mu = 1. / N * np.sum(input_tensor, axis=0)

            # step2: subtract mean vector of every trainings example
            self.xmu = input_tensor - self.mu

            # step3: following the lower branch - calculation denominator
            sq = self.xmu ** 2

            # step4: calculate variance
            self.var = 1. / N * np.sum(sq, axis=0)

            # step5: add eps for numerical stability, then sqrt
            self.sqrtvar = np.sqrt(self.var + self.epsilon)

            # step6: invert sqrtwar
            self.ivar = 1. / self.sqrtvar

            # step7: execute normalization
            self.xhat = self.xmu * self.ivar

            # step8: Nor the two transformation steps
            self.gammax = self.weights * self.xhat

            # step9
            self.forward_output = self.gammax + self.bias

            ######
            #self.mean           = np.mean(input_tensor, axis=0)
            #self.var            = np.var(input_tensor, axis=0)
            #self.forward_output = np.zeros_like(input_tensor)
            #self.forward_output = (input_tensor - self.mean) / (np.sqrt( self.var  + self.epsilon ) )
            #self.forward_output = self.forward_output * self.weights + self.bias

        if self.phase == Base.Phase.test:
            self.forward_output = np.zeros_like(input_tensor)
            self.forward_output = (input_tensor - self.mu) / (np.sqrt(self.var + self.epsilon))
            self.forward_output = self.forward_output * self.weights + self.bias

        if self.channels > 0:
            self.forward_output = self.forward_output.reshape(shapeBevorReshaping)
            self.forward_output = np.transpose(self.forward_output, (0,3,2,1))

        return self.forward_output

    def backward(self, error_tensor):
        IsConvInput = False
        input_tensor = self.input_tensor
        if error_tensor.ndim == 4:
            error_tensor = np.transpose(error_tensor, (0, 3, 2, 1))
            shapeBevorReshaping = error_tensor.shape
            error_tensor = error_tensor.reshape(-1, error_tensor.shape[3])

            input_tensor = np.transpose(input_tensor, (0, 3, 2, 1))
            shapeBevorReshaping = input_tensor.shape
            input_tensor = input_tensor.reshape(-1, input_tensor.shape[3])
            IsConvInput = True

        xhat = self.xhat
        gamma = self.weights
        xmu = self.xmu
        ivar = self.ivar
        sqrtvar = self.sqrtvar
        var = self.var
        eps = self.epsilon
        dout = error_tensor

        # get the dimensions of the input/output
        N, D = dout.shape

        # step9
        self.dbeta = np.sum(dout, axis=0)
        dgammax = dout  # not necessary, but more understandable

        # step8
        self.dgamma = np.sum(dgammax * xhat, axis=0)
        dxhat = dgammax * gamma

        # step7
        divar = np.sum(dxhat * xmu, axis=0)
        dxmu1 = dxhat * ivar

        # step6
        dsqrtvar = -1. / (sqrtvar ** 2) * divar

        # step5
        dvar = 0.5 * 1. / np.sqrt(var + eps) * dsqrtvar

        # step4
        dsq = 1. / N * np.ones((N, D)) * dvar

        # step3
        dxmu2 = 2 * xmu * dsq

        # step2
        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)

        # step1
        dx2 = 1. / N * np.ones((N, D)) * dmu

        # step0
        dx = dx1 + dx2

        if IsConvInput == True:
            dx = dx.reshape(shapeBevorReshaping)
            dx = np.transpose(dx, (0,3,2,1))

        #update
        if hasattr(self, 'optimizer'):
            self.weights = self.optimizer.calculate_update(1, self.weights, self.dgamma)
            self.bias = self.optimizer.calculate_update(1, self.bias, self.dbeta)

        return dx


    def get_gradient_bias(self):
        return self.dbeta

    def get_gradient_weights(self):
        return self.dgamma

    def initialize(self, weights_initializer, bias_initializer):
        self.weightsInitializer = weights_initializer
        self.biasInitializer    = bias_initializer
        self.needForInit        = True