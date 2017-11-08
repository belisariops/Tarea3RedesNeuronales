from abc import ABC, abstractmethod

from src.SigmoidNeuron import SigmoidNeuron


class AbstractNeuralLayer(ABC):
    def __init__(self, neuron_array=None):
        self.neuron_array = neuron_array
        if neuron_array is None:
            self.neuron_array = []
        self.next_layer = None
        self.previous_layer = None

    def buildRandomLayer(self, number_of_neurons):
        neuron = None

        for i in range(number_of_neurons):
            neuron = SigmoidNeuron()
            neuron.setRandomParameters()
            self.neuron_array.append(neuron)

    def setLearningRate(self, learning_rate):
        for neuron in self.neuron_array:
            neuron.setC(learning_rate)
        self.next_layer.setLearningRate(learning_rate)

    def forwardPropagation(self, inputs):
        outputs = []
        for neuron in self.neuron_array:
            neuron.updateWeights(inputs)
            neuron.updateBias()
            outputs.append(neuron.output)
        self.next_layer.forwardPropagation(outputs)

    def transferDerivative(self, output):
        return output * (1.0 - output)

    def setPreviousLayer(self, previous_layer):
        self.previous_layer = previous_layer

    def setNextLayer(self, next_layer):
        self.next_layer = next_layer

    def getNumberofNeurons(self):
        return len(self.neuron_array)

    @abstractmethod
    def getOutputs(self, inputs):
        pass

    def setRandomWeights(self, number_of_weights, min_value, max_value):
        for neuron in self.neuron_array:
            neuron.setRandomWeights(number_of_weights, min_value, max_value)

    def calculateDelta(self, expected_output):
        for index in range(len(self.neuron_array)):
            error = 0
            for next_neuron in self.next_layer.neuron_array:
                error += next_neuron.weights[index] * next_neuron.delta
            self.neuron_array[index].delta = error * self.transferDerivative(self.neuron_array[index].output)

    def buildFromArray(self, neuron_array):
        """
        Builds a neural layer from an array of neurons weights and bias.
        :param array of serialized values of a neuron:
        """
        # In neuron values the las value is the bias of the neuron
        for neuron_values in neuron_array:
            neuron = SigmoidNeuron()
            #Get all except the last one
            weights = neuron_values[:-1]
            #Get only the last one
            bias = neuron_values[-1]
            neuron.weights = weights
            neuron.bias = bias
            self.neuron_array.append(neuron)
