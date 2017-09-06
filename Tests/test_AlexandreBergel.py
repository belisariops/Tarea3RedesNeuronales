import unittest
from src.FirstNeuralLayer import FirstNeuralLayer
from src.LastNeuralLayer import LastNeuralLayer
from src.NeuralNetwork import NeuralNetwork
from src.SigmoidNeuron import SigmoidNeuron


class MyTestCase(unittest.TestCase):
    def test_network_one_epoch(self):
        first_layer = FirstNeuralLayer()
        output_layer = LastNeuralLayer()
        hidden1 = SigmoidNeuron()
        hidden1.weights = [0.1,0.2]
        hidden1.setBias(0.1)
        hidden2 = SigmoidNeuron()
        hidden2.weights = [0.2,0.3]
        hidden2.setBias(0.1)
        hidden1.setC(0.5)
        hidden2.setC(0.5)

        out = SigmoidNeuron()
        out.weights = [0.3,0.4]
        out.setBias(0.1)
        out.setC(0.5)
        first_layer.neuron_array = [hidden1, hidden2]
        output_layer.neuron_array = [out]

        first_layer.setNextLayer(output_layer)
        output_layer.setPreviousLayer(first_layer)


        neural_network = NeuralNetwork(2)
        neural_network.first_layer = first_layer
        neural_network.output_layer = output_layer

        neural_network.train(1,[[0.9,0.8,[1]]])
        print(hidden1.output)

        #Network weights
        self.assertEqual(hidden1.weights[0],0.1028369848921488)
        self.assertEqual(hidden1.weights[1],0.20252176434857672)
        self.assertEqual(hidden2.weights[0],0.20364750008778296)
        self.assertEqual(hidden2.weights[1],0.3032422223002515)
        self.assertEqual(hidden2.weights[1],0.3032422223002515)
        self.assertEqual(out.weights[0],0.3254179929201722)
        self.assertEqual(out.weights[1],0.42717415579920276)

        #Network biases
        self.assertEqual(hidden1.bias,0.10315220543572089)
        self.assertEqual(hidden1.bias,0.10315220543572089)
        self.assertEqual(hidden1.bias,0.10315220543572089)
        self.assertEqual(out.bias,0.14332974979557217)

        #Network deltas
        self.assertEqual(hidden1.delta,0.006304410871441775)
        self.assertEqual(hidden2.delta,0.008105555750628777)
        self.assertEqual(out.delta,0.08665949959114436)

        #Network outputs
        self.assertEqual(hidden1.output, 0.5866175789173301)
        self.assertEqual(hidden2.output, 0.6271477663131956)
        self.assertEqual(out.output, 0.6287468135085144)






if __name__ == '__main__':
    unittest.main()
