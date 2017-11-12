import random

import matplotlib.pylab as plt
import numpy as np
from numpy import linalg as LA

from src.FirstNeuralLayer import FirstNeuralLayer
from src.InnerNeuralLayer import InnerNeuralLayer
from src.LastNeuralLayer import LastNeuralLayer

from src.SigmoidNeuron import SigmoidNeuron


class NeuralNetwork:
    def __init__(self, numberOfInputs):
        self.first_layer = None
        self.output_layer = None
        self.min_neurons_per_layer = 0
        self.max_neurons_per_layer = 0
        self.numberOfInputs = numberOfInputs
        self.error = 0
        self.error_plotX = []
        self.error_plotY = []
        self.precisionX = []
        self.precisionY = []
        self.number_of_layers = 0

    def buildLayer(self, layer, num_neurons):
        for index in range(num_neurons):
            neuron = SigmoidNeuron()
            layer.neuron_array.append(neuron)

    def buildFixed(self, layers):
        if len(layers) < 1:
            raise ValueError('La red no tiene ningun elemento')

        first_layer = FirstNeuralLayer()
        self.buildLayer(first_layer, layers[0])
        last_layer = LastNeuralLayer()
        self.buildLayer(last_layer, layers[-1])
        current_layer = first_layer
        self.number_of_layers += 2
        for num_neurons in layers[1:-1]:
            inner_layer = InnerNeuralLayer()
            self.number_of_layers += 1
            self.buildLayer(inner_layer, num_neurons)
            current_layer.setNextLayer(inner_layer)
            inner_layer.setPreviousLayer(current_layer)
            current_layer = inner_layer
        current_layer.setNextLayer(last_layer)
        last_layer.setPreviousLayer(current_layer)
        self.first_layer = first_layer
        self.output_layer = last_layer

    def setLearningRate(self, learning_rate):
        self.first_layer.setLearningRate(learning_rate)

    def createLayer(self, neural_layer, previous_layer):
        neural_layer.buildRandomLayer(random.randint(self.min_neurons_per_layer, self.max_neurons_per_layer))
        if previous_layer is None:
            first_layer = neural_layer
            number_weights = self.numberOfInputs
        else:
            previous_layer.setNextLayer(neural_layer)
            number_weights = previous_layer.getNumberofNeurons()
        neural_layer.setRandomWeights(number_weights, -1, 2)
        neural_layer.setPreviousLayer(previous_layer)
        previous_layer = neural_layer
        return neural_layer

    def setRandomLayers(self, number_of_layers, min_neurons_per_layer, max_neurons_per_layer, number_of_outputs):
        first_layer = FirstNeuralLayer()
        neural_layer = first_layer
        self.min_neurons_per_layer = min_neurons_per_layer
        self.max_neurons_per_layer = max_neurons_per_layer
        previous_layer = self.createLayer(first_layer, None)
        for i in range(number_of_layers - 1):
            neural_layer = InnerNeuralLayer()
            previous_layer = self.createLayer(neural_layer, previous_layer)

        neural_layer = LastNeuralLayer()
        # neural_layer = self.createLayer(neural_layer,previous_layer)
        neural_layer.buildRandomLayer(number_of_outputs)
        neural_layer.setPreviousLayer(previous_layer)
        previous_layer.setNextLayer(neural_layer)
        neural_layer.setRandomWeights(len(neural_layer.previous_layer.neuron_array), -1, 2)

        self.output_layer = neural_layer

        self.first_layer = first_layer

    def setInputs(self, inputs):
        self.inputs = inputs

    def addLayer(self, neural_layer):
        self.first_layer.append(neural_layer)

    def addRandomLayer(self, number_of_neurons):
        self.first_layer.append(InnerNeuralLayer().buildRandomLayer(number_of_neurons))

    def feed(self, inputs):
        return self.first_layer.getOutputs(inputs)

    # def getOutput(self,inputs):

    def addLastLayer(self):
        layer = self.first_layer
        while layer is not None:
            current_layer = layer
            layer = layer.next_layer
        last_layer = LastNeuralLayer()
        last_layer.buildRandomLayer(1)
        last_layer.setRandomWeights(current_layer.getNumberofNeurons(), -3, 3)
        current_layer.setNextLayer(last_layer)
        last_layer.setPreviousLayer(current_layer)
        self.output_layer = last_layer

    def forwardPropagation(self, input):
        self.first_layer.forwardPropagation(input)

    def train(self, numberOfEpochs, data, test_data=None):
        for i in range(numberOfEpochs):
            self.error = 0
            for set in data:
                input_data = set[0:self.numberOfInputs]
                output_last_layer = self.feed(input_data)
                expected_output = set[-1:][0]
                self.output_layer.backPropagation(expected_output)
                self.forwardPropagation(input_data)
                # print(expected_output)
                # print(output_last_layer)
                self.error += (np.power(LA.norm(np.subtract(expected_output, output_last_layer)), 2) / len(data))
            # In the tests this has to be skipped
            if test_data is not None:
                x = self.getGuessRatio(test_data)
                self.precisionY.append(x)
                self.precisionX.append(i)
                self.error_plotX.append(i)
                self.error_plotY.append(self.error)

                # error = expected_output - output_last_layer
                # delta = error * (output_last_layer * (1.0 - output_last_layer))

    def plotErrorData(self):
        plt.figure()
        plt.title("Precision", fontsize=20)
        plt.xlabel('epochs')
        plt.ylabel('ratio')
        plt.plot(self.precisionX, self.precisionY)
        plt.figure()
        plt.title("Error", fontsize=20)
        plt.xlabel('epochs')
        plt.ylabel('error')
        plt.plot(self.error_plotX, self.error_plotY)
        plt.show()

    def getGuessRatio(self, test_data):
        total = float(len(test_data))
        correct_guesses = 0
        for data in test_data:
            guess = self.interp(self.feed(data[0:len(data)]))
            if guess == data[len(data) - 1]:
                correct_guesses += 1
        return correct_guesses

    def interp(self, output):
        index = -1
        max = -1
        for option in output:
            if option > max:
                max = option
                index += 1
        resp = []
        for i in range(self.output_layer.getNumberofNeurons()):
            resp.append(0)
        resp[index] = 1
        return resp

    def load_network(self, serialize_network):
        """
        From an array the layers, weight and bias are created.
        """
        layer = self.first_layer
        current_index = 0
        for i in range(self.number_of_layers):
            for neuron in layer.neuron_array:
                number_of_weights = len(neuron.weights)
                neuron.weights = serialize_network[current_index: number_of_weights]
                neuron.bias = serialize_network[current_index + number_of_weights]
                current_index += number_of_weights + 1
            if i != self.number_of_layers - 1:
                layer = layer.next_layer
