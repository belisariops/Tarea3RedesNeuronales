from src.AbstractNeuralLayer import AbstractNeuralLayer



class FirstNeuralLayer(AbstractNeuralLayer):

    def backPropagation(self, expected_output):
        self.calculateDelta(expected_output)

    def getOutputs(self, inputs):
        outputs = []
        for neuron in self.neuron_array:
            #print(inputs)
            outputs.append(neuron.getOutput(inputs))
        return self.next_layer.getOutputs(outputs)