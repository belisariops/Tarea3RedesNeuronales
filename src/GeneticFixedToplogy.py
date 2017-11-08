import random
import sys

from src.AbstractGeneticAlgorithm import AbstractGeneticAlgorithm
from src.FileManager import FileManager
from src.NeuralNetwork import NeuralNetwork


class GeneticFixedTopology(AbstractGeneticAlgorithm):
    def __init__(self, initial_population, expected_error):
        """
        A genetic algorithm is used to learn the weights and bias of a topology
        fixed network.
        """
        super().__init__(initial_population)
        self.expected_error = expected_error
        self.error = sys.maxsize
        self.epoch = 0
        self.neurons_per_layer = [3, 4, 3]
        self.num_inputs = 4
        # Build Fixed Neural Network, with 4 inputs
        self.neural_network = NeuralNetwork(self.num_inputs)
        # The neural network has 3 layers with 3,4 and 3 neurons in each
        self.neural_network.buildFixed(self.neurons_per_layer)
        self.test_values = 50
        # Parse data set
        file_manager = FileManager()
        file_manager.load_file("../Datasets/iris.data")
        self.train_data = file_manager.get_train_data()
        self.test_data = file_manager.get_test_data()

    def run(self):
        """
        Execute the genetic algorithm to find the best weights and bias
        for fixed neural network.
        """
        self.initialize_population(self.initial_population)
        while self.error > self.expected_error:
            self.generation += 1
            self.selection()

    def reproduction(self):
        """
        Reproduce the population.
        """
        pass

    def create_random_bias(self):
        return random.uniform(1.0, 3.0)

    def create_random_weight(self):
        return random.uniform(0.0, 2.0)

    def initialize_population(self, number_of_individuals):
        """
        Creates a fixed number of individuals with the same properties,
        but different weights and bias.
        """
        num_inputs = self.num_inputs
        for num_neurons in self.neurons_per_layer:
            self.number_genes += (num_neurons*num_inputs + 1)

        # for i in range(number_of_individuals):
        #     layer = []
        #     num_inputs = self.num_inputs
        #     for num_neurons in self.neurons_per_layer:
        #         for index in range(num_inputs):
        #             layer.append(self.create_random_weight())
        #         layer.append(self.create_random_bias())
        #     self.population.append(layer)

    def selection(self):
        """
        Select a fixed number of suitable candidates.
        """
        pass

    def evaluate_fitness(self, individual):
        """
        Returns the fitness value of an individual.
        """
        fitness = 0
        self.neural_network.load_network(individual)
        for i in range(self.test_values):
            data = self.test_data[i]
            correct_result = data[-1]
            raw_result = self.neural_network.feed(data[0:-1])
            guess_result = self.neural_network.interp(raw_result)
            if correct_result == guess_result:
                fitness += 1
        return fitness
