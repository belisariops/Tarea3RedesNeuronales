from AbstractNeuralLayer import AbstractNeuralLayer


class LastNeuralLayer(AbstractNeuralLayer):

    def backPropagation(self, expected_output):
        index = 0
        for neuron in self.neuron_array:
            error = expected_output[index] - neuron.output
            neuron.delta = error * self.transferDerivative(neuron.output)
            index += 1
        self.previous_layer.backPropagation(expected_output)

    def getOutputs(self, inputs):
        outputs = []
        for neuron in self.neuron_array:
            neuron.output = neuron.getOutput(inputs)
            outputs.append(neuron.output)

        return outputs

    def forwardPropagation(self,inputs):
        for neuron in self.neuron_array:
            neuron.updateWeights(inputs)
            neuron.updateBias()

    def setLearningRate(self,learning_rate):
        for neuron in self.neuron_array:
            neuron.setC(learning_rate)
