import random
import sys

from src.AbstractGeneticAlgorithm import AbstractGeneticAlgorithm
from src.FileManager import FileManager
from src.NeuralNetwork import NeuralNetwork


class GeneticFixedTopology(AbstractGeneticAlgorithm):
    def __init__(self, initial_population, expected_precision):
        """
        A genetic algorithm is used to learn the weights and bias of a topology
        fixed network.
        """
        super().__init__(initial_population)
        self.expected_precision = expected_precision
        self.precision = 0
        self.epoch = 0
        self.num_inputs = 4
        self.neurons_per_layer = [self.num_inputs, 5, 3]
        # Build Fixed Neural Network, with 4 inputs
        self.neural_network = NeuralNetwork(self.num_inputs)
        # The neural network has 3 layers with 3,4 and 3 neurons in each
        self.neural_network.buildFixed(self.neurons_per_layer)
        self.test_values = 100
        # Parse data set
        file_manager = FileManager()
        file_manager.load_file("../Datasets/iris.data")
        self.train_data = file_manager.get_train_data()
        self.test_data = file_manager.get_test_data()
        self.neurons_position = []

    def run(self):
        """
        Execute the genetic algorithm to find the best weights and bias
        for fixed neural network.
        """
        self.initialize_population(self.population_size)
        while self.precision < self.expected_precision:
            print(self.precision)
            self.generation += 1
            child_population = []
            while len(child_population) < self.population_size:
                father = self.selection()
                mother = self.selection()
                first_child, second_child = self.cross_over(father, mother)
                self.mutate(first_child)
                self.mutate(second_child)
                if self.evaluate_fitness(first_child) >= self.evaluate_fitness(second_child):
                    child_population.append(first_child)
                else:
                    child_population.append(second_child)
            self.population = child_population

        return self.get_best_neural_network()

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
            self.number_genes += ((num_neurons + 1) * num_inputs)

        super().initialize_population(number_of_individuals)

        # for i in range(number_of_individuals):
        #     layer = []
        #     num_inputs = self.num_inputs
        #     for num_neurons in self.neurons_per_layer:
        #         for index in range(num_inputs):
        #             layer.append(self.create_random_weight())
        #         layer.append(self.create_random_bias())
        #     self.population.append(layer)

    def evaluate_fitness(self, individual):
        """
        Returns the fitness value of an individual.
        """
        fitness = 0
        self.neural_network.load_network(individual)
        for i in range(self.test_values):
            data = self.test_data[random.randint(0, len(self.test_data) - 1)]
            correct_result = data[-1]
            raw_result = self.neural_network.feed(data[0:-1])
            guess_result = self.neural_network.interp(raw_result)
            if correct_result == guess_result:
                fitness += 1
        ratio = float(fitness / self.test_values)
        if ratio > self.precision:
            self.precision = ratio
        return fitness

    def get_best_neural_network(self):
        best_individual = self.population[0]
        best_fitness = self.evaluate_fitness(best_individual)
        for serialized_netowrk in self.population:
            individual_fitness = self.evaluate_fitness(serialized_netowrk)
            if best_fitness < individual_fitness:
                best_individual = serialized_netowrk
                best_fitness = individual_fitness
        return self.neural_network.load_network(best_individual)

    def mutate(self, individual):
        for index in range(len(self.neurons_position)):
            if self.mutation_rate > random.uniform(0, 1):
                if index != len(self.neurons_position) - 1:
                    dif = self.neurons_position[index + 1] - self.neurons_position[index]
                else:
                    dif = len(individual) - 1 - self.neurons_position[index]
                for j in range(dif):
                    individual[index + j] = random.uniform(-100.0, 100.0)


