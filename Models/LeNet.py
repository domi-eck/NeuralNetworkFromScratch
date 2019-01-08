from Layers import *
from Optimization import *
import NeuralNetwork

def build():
    optimizer   = Optimizers.Adam(0.0005)
    optimizer.add_regularizer(Constraints.L2_Regularizer(0.0004))
    weightInit  = Initializers.Xavier()
    baisInit    = Initializers.Constant(0)
    network     = NeuralNetwork.NeuralNetwork(optimizer, weightInit, baisInit)
    network.loss_layer = SoftMax.SoftMax()
    network.layers.append(BatchNormalization.BatchNormalization(1))
    network.append_trainable_layer(Conv.Conv([1,1],[1,5,5],6))
    network.layers.append(ReLU.ReLU())
    network.layers.append(Dropout.Dropout(0.5))
    network.layers.append(Pooling.Pooling([2,2],[2,2]))
    network.layers.append(BatchNormalization.BatchNormalization(6))
    network.append_trainable_layer(Conv.Conv([1,1],[6,5,5],16))
    network.layers.append(ReLU.ReLU())
    network.layers.append(Dropout.Dropout(0.7))
    network.layers.append(Pooling.Pooling([2,2],[2,2]))
    network.layers.append(Flatten.Flatten())
    network.layers.append(BatchNormalization.BatchNormalization())
    network.append_trainable_layer(FullyConnected.FullyConnected(7 * 7 * 16, 120))
    network.layers.append(ReLU.ReLU())
    network.layers.append(Dropout.Dropout(0.8))
    network.layers.append(BatchNormalization.BatchNormalization())
    network.append_trainable_layer(FullyConnected.FullyConnected(120, 84))
    network.layers.append(ReLU.ReLU())
    network.layers.append(Dropout.Dropout(0.9))
    network.layers.append(BatchNormalization.BatchNormalization())
    network.append_trainable_layer(FullyConnected.FullyConnected(84, 10))
    network.layers.append(ReLU.ReLU())
    return network

