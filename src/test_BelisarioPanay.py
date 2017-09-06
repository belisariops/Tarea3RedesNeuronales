import unittest

from FileManager import normalize, FileManager
from FirstNeuralLayer import FirstNeuralLayer
from LastNeuralLayer import LastNeuralLayer
from NeuralNetwork import NeuralNetwork
from Parser import Parser
from SigmoidNeuron import SigmoidNeuron


class MyTestCase(unittest.TestCase):
    def test_neuron(self):
        neuron = SigmoidNeuron()
        neuron.weights = [0.5, 0.5]
        neuron.setBias(0)
        self.assertEqual(neuron.getOutput([1, -1]), 0.5)

    def build_network(self):
        network = NeuralNetwork(2)
        neuron_1 = SigmoidNeuron()
        neuron_2 = SigmoidNeuron()
        neuron_3 = SigmoidNeuron()
        neuron_1.weights = [0.3,0.6]
        neuron_2.weights = [1.2,-0.6]
        neuron_3.weights = [0.7,0.6]

        neuron_4 = SigmoidNeuron()
        neuron_5 = SigmoidNeuron()
        neuron_4.weights = [0.5,0.2,1.1]
        neuron_5.weights = [-0.5,0.5,0.5]

        first_layer = FirstNeuralLayer()
        first_layer.neuron_array = [neuron_1, neuron_2, neuron_3]
        output_layer = LastNeuralLayer()
        output_layer.neuron_array = [neuron_4, neuron_5]
        first_layer.setNextLayer(output_layer)
        output_layer.setPreviousLayer(first_layer)
        network.first_layer = first_layer
        network.output_layer = output_layer
        return network

    def test_output_interpreter(self):
        network = self.build_network()
        interpret_output1 = network.interp([0.023, 0.99])
        interpret_output2 = network.interp([0.8, 0.1])
        interpret_output3 = network.interp([0.49, 0.5])
        interpret_output4 = network.interp([0.51, 0.5])
        self.assertEqual(interpret_output1, [0, 1])
        self.assertEqual(interpret_output2, [1, 0])
        self.assertEqual(interpret_output3, [0, 1])
        self.assertEqual(interpret_output4, [1, 0])

    def test_OR(self):
        network = self.build_network()
        data = [[0, 0, [1,0]], [0, 1,[0,1]], [1, 0,[0,1]], [1, 1, [0,1]]]  # OR data
        network.train(2000, data)
        output = network.feed([0, 1, [0, 1]])
        interpreter_output = network.interp(output)
        self.assertEqual(interpreter_output,[0,1])

    def test_XOR(self):
        network = self.build_network()
        data = [[0, 0, [1, 0]], [0, 1, [0, 1]], [1, 0, [0, 1]], [1, 1, [1, 0]]]  # XOR data
        network.train(2000, data)
        output = network.feed([1, 1, [1, 0]])
        interpreter_output = network.interp(output)
        self.assertEqual(interpreter_output, [1, 0])

    def test_iris_parser(self):
        parser = Parser()
        parsed_data= parser.parse_iris_data(["1,2,3,4,Iris-versicolor"])
        self.assertEqual([[1.0, 2.0, 3.0, 4.0, [0, 1, 0]]],parsed_data)

    def test_normalization(self):
        test_array = [8,8,8]
        normalize_test_array = list(map(normalize,test_array))
        self.assertEqual(normalize_test_array,[1,1,1])



    def test_file_manager(self):
        file_manager = FileManager()
        file_manager.load_file("../Datasets/test.data")
        normalize_data_1 = [[2.0,2.0,2.0,2.0,[0,1,0]]]
        normalize_data_2 = [[1.0,1.0,1.0,1.0,[1,0,0]]]
        self.assertEqual(file_manager.get_train_data(),normalize_data_1)
        self.assertEqual(file_manager.get_test_data(),normalize_data_2)


if __name__ == '__main__':
    unittest.main()
