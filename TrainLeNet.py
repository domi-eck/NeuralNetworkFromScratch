from Layers import Helpers
from Models.LeNet import build
import NeuralNetwork
import matplotlib.pyplot as plt
import os.path

batch_size = 50
mnist = Helpers.MNISTData(batch_size)
#mnist.show_random_training_image()

if os.path.isfile('trained/LeNet_alex_300epo_vielDropout'):
    net = NeuralNetwork.load('trained/LeNet_alex_300epo_vielDropout', mnist)
else:
    net = build()
    net.data_layer = mnist

i = 0
while (i < 10):
    i += 1
    net.train(5)
    data, labels = net.data_layer.get_test_set()
    data = data[0:50]
    labels = labels[0:50]
    results = net.test(data)

    accuracy = Helpers.calculate_accuracy(results, labels)
    print('\nOn the MNIST dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')

NeuralNetwork.save('trained/LeNet_alex_300epo_vielDropout', net)

#plt.figure('Loss function for training LeNet_alex_300epo_vielDropout on the MNIST dataset')
#plt.plot(net.loss, '-x')
#plt.show()

data, labels = net.data_layer.get_test_set()
data = data [0:50]
labels = labels [0:50]
results = net.test(data)

accuracy = Helpers.calculate_accuracy(results, labels)
print('\nOn the MNIST dataset, we achieve an accuracy of: ' + str(accuracy * 100) + '%')