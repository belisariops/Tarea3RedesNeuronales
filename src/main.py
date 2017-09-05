import random

from src.FileManager import FileManager
from src.FirstNeuralLayer import FirstNeuralLayer
from src.LastNeuralLayer import LastNeuralLayer

from src.NeuralNetwork import NeuralNetwork
from src.SigmoidNeuron import SigmoidNeuron


def main():
    file_manager = FileManager()
    file_manager.load_file("../Datasets/iris.data")
    train_data = file_manager.get_train_data()
    neural_network = NeuralNetwork(4)
    neural_network.setRandomLayers(1, 4, 4, 3)
    # first_layer = FirstNeuralLayer()
    # output_layer = LastNeuralLayer()
    # hidden1 = SigmoidNeuron()
    # hidden1.weights = [0.1,0.2]
    # hidden1.setBias(0.1)
    # hidden2 = SigmoidNeuron()
    # hidden2.weights = [0.2,0.3]
    # hidden2.setBias(0.1)
    #
    # out = SigmoidNeuron()
    # out.weights = [0.3,0.4]
    # out.setBias(0.1)
    # first_layer.neuron_array = [hidden1, hidden2]
    # output_layer.neuron_array = [out]
    #
    # first_layer.setNextLayer(output_layer)
    # output_layer.setPreviousLayer(first_layer)
    #
    #
    # neural_network = NeuralNetwork(2)
    # neural_network.first_layer = first_layer
    # neural_network.output_layer = output_layer
    #
    # neural_network.train(1,[[0.9,0.8,[1]]],[[0,1,[0]]])

    # first_layer = FirstNeuralLayer()
    # learningRate = 0.1
    # bias = random.uniform(1, 3)
    # neuron_0 = SigmoidNeuron()
    # neuron_0.weights = [1, 1.1,2.3,-0.6]
    # neuron_0.setC(learningRate)  # 0.2)
    # neuron_0.setBias(bias)
    # bias = random.uniform(1, 3)
    # neuron_1 = SigmoidNeuron()
    # neuron_1.weights = [-1, 2,1.3,0.6]
    # neuron_1.setC(learningRate)  # 4#)
    # neuron_1.setBias(bias)
    # neuron_2 = SigmoidNeuron()
    # neuron_2.weights = [1, 2.3,1,-0.3]
    # neuron_2.setC(learningRate)  # 8)
    # bias = random.uniform(1, 3)
    # neuron_2.setBias(bias)
    #
    # neuron_3 = SigmoidNeuron()
    # neuron_3.weights = [-1.2, 1.3,2.2,0.4]
    # neuron_3.setC(learningRate)  # 8)
    # bias = random.uniform(1, 3)
    # neuron_3.setBias(bias)
    #
    # neuron_4 = SigmoidNeuron()
    # neuron_4.weights = [1, 1.5,-1,3]
    # neuron_4.setC(learningRate)  # 8)
    # bias = random.uniform(1, 3)
    # neuron_4.setBias(bias)
    #
    # neuron_5 = SigmoidNeuron()
    # neuron_5.weights = [-2, 2.3, 1.2,1.1]
    # neuron_5.setC(learningRate)  # 8)
    # bias = random.uniform(1, 3)
    # neuron_5.setBias(bias)
    #
    # neuron_6 = SigmoidNeuron()
    # neuron_6.weights = [1.4, 2.5, -1.2,0.3]
    # neuron_6.setC(learningRate)  # 8)
    # bias = random.uniform(1, 3)
    # neuron_6.setBias(bias)
    #
    # neuron_7 = SigmoidNeuron()
    # neuron_7.weights = [0.2,-0.6,0.4, 1.5, -2.2, -1.8, -0.3]
    # neuron_7.setC(learningRate)  # 8)
    # bias = random.uniform(1, 3)
    # neuron_7.setBias(bias)
    #
    # neuron_8 = SigmoidNeuron()
    # neuron_8.weights = [1.4, 2.5,1.3,1, -1.2, 1.8, 0.3]
    # neuron_8.setC(learningRate)  # 8)
    # bias = random.uniform(1, 3)
    # neuron_8.setBias(bias)
    #
    # neuron_9 = SigmoidNeuron()
    # neuron_9.weights = [0.4, 1.5, -2.2, -1.8,2.2,-0.1, -0.3]
    # neuron_9.setC(learningRate)  # 8)
    # bias = random.uniform(1, 3)
    # neuron_9.setBias(bias)
    #
    #
    # first_layer.neuron_array = [neuron_0, neuron_1,neuron_2,neuron_3,neuron_4,neuron_5,neuron_6]
    # last_layer = LastNeuralLayer()
    # last_layer.neuron_array = [neuron_7,neuron_8,neuron_9]
    # first_layer.setNextLayer(last_layer)
    # last_layer.setPreviousLayer(first_layer)
    # neural_network.first_layer = first_layer
    # neural_network.output_layer = last_layer
    # myfunc = lambda x, y: 1 if (x != y) else 0
    # output = neural_network.feed([1, 1])
    #print(output)
    test_data = file_manager.get_test_data()
    neural_network.train(300, train_data,test_data)
    #print(myfunc(0, 0))

    output = neural_network.feed(test_data[0][0:5])
    print(output)
    print("Test data ratio: {0}".format(neural_network.getGuessRatio(test_data)))
    neural_network.plotErrorData()
    x =2



main()
