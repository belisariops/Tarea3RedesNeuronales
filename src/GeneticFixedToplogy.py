import random
import sys
import matplotlib.pylab as plt

from src.AbstractGeneticAlgorithm import AbstractGeneticAlgorithm
from src.FileManager import FileManager
from src.NeuralNetwork import NeuralNetwork


class GeneticFixedTopology(AbstractGeneticAlgorithm):
    def __init__(self, initial_population,generations):
        """
        A genetic algorithm is used to learn the weights and bias of a topology
        fixed network.
        """
        super().__init__(initial_population)
        #self.expected_precision = expected_precision
        self.generation_span = generations
        self.precision = 0
        self.epoch = 0
        self.num_inputs = 4
        self.neurons_per_layer = [self.num_inputs, 4, 3]
        # Build Fixed Neural Network, with 4 inputs
        self.neural_network = NeuralNetwork(self.num_inputs)
        # The neural network has 3 layers with 3,4 and 3 neurons in each
        self.neural_network.buildFixed(self.neurons_per_layer)
        self.test_values = 20
        # Parse data set
        file_manager = FileManager()
        file_manager.load_file("../Datasets/iris.data")
        self.train_data = file_manager.get_train_data()
        self.test_data = file_manager.get_test_data()
        self.neurons_position = []
        self.x_plot = []
        self.y_plot = []

    def run(self):
        """
        Execute the genetic algorithm to find the best weights and bias
        for fixed neural network.
        """
        self.initialize_population(self.population_size)
        while self.generation < self.generation_span:
            self.precision = 0
            self.generation += 1
            self.x_plot.append(self.generation)
            child_population = []
            while len(child_population) < self.population_size:
                father = self.selection()
                mother = self.selection()
                first_child, second_child = self.cross_over(father, mother)
                self.mutate(first_child)
                self.mutate(second_child)
                father_fitness = self.evaluate_fitness(father)
                mother_fitness = self.evaluate_fitness(mother)
                fitness_first_child = self.evaluate_fitness(first_child)
                fitness_second_child = self.evaluate_fitness(second_child)

                if fitness_first_child >= fitness_second_child:
                    if fitness_first_child >= father_fitness and fitness_first_child >= mother_fitness:
                        child_population.append(first_child)
                        self.set_best_new_population_fitness(fitness_first_child)
                    else:
                        if father_fitness > mother_fitness:
                            child_population.append(father)
                            self.set_best_new_population_fitness(father_fitness)
                        else:
                            child_population.append(mother)
                            self.set_best_new_population_fitness(mother_fitness)

                else:
                    if fitness_second_child >= father_fitness and fitness_second_child >= mother_fitness:
                        child_population.append(second_child)
                        self.set_best_new_population_fitness(fitness_second_child)
                    else:
                        if father_fitness > mother_fitness:
                            child_population.append(father)
                            self.set_best_new_population_fitness(father_fitness)
                        else:
                            child_population.append(mother)
                            self.set_best_new_population_fitness(mother_fitness)

            self.y_plot.append(self.precision)
            self.population = child_population

        return self.get_best_neural_network()

    def set_best_new_population_fitness(self, fitness):
        ratio = float(fitness / self.test_values)
        if ratio > self.precision:
            self.precision = ratio

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
        position = 0
        #Input layer
        for initial_neuron in range(self.num_inputs):
            self.neurons_position.append(position)
            position +=2

        i = 1
        for num_neurons in self.neurons_per_layer[1:]:
            for j in range(num_neurons):
                self.neurons_position.append(position)
                position += (self.neurons_per_layer[i - 1] + 1)
            i += 1
        self.number_genes = position
        super().initialize_population(number_of_individuals)

    def evaluate_fitness(self, individual):
        """
        Returns the fitness value of an individual.
        """
        fitness = 0
        if individual.fitness is None:
            self.neural_network.load_network(individual.serialized_values)
            for i in range(self.test_values):
                data = self.test_data[random.randint(0, len(self.test_data) - 1)]
                correct_result = data[-1]
                raw_result = self.neural_network.feed(data[0:-1])
                guess_result = self.neural_network.interp(raw_result)
                if correct_result == guess_result:
                    fitness += 1
            individual.fitness = fitness
        return individual.fitness

    def get_best_neural_network(self):
        best_individual = self.population[0]
        if best_individual.fitness is None:
            self.evaluate_fitness(best_individual)
        for individual in self.population:
            if individual.fitness is None:
                self.evaluate_fitness(individual)
            if best_individual.fitness < individual.fitness:
                best_individual = individual
        return self.neural_network.load_network(best_individual.serialized_values)

    def mutate(self, individual):
        for index in range(len(self.neurons_position)):
            if self.mutation_rate > random.uniform(0, 1):
                if index != len(self.neurons_position) - 1:
                    dif = self.neurons_position[index + 1] - self.neurons_position[index]
                else:
                    dif = len(individual.serialized_values) - 1 - self.neurons_position[index]
                for j in range(dif):
                    individual.serialized_values[index + j] = random.uniform(-15.0, 15.0)

    def plot_results(self):
        plt.figure()
        plt.title("Precisión", fontsize=20)
        plt.xlabel('genercación')
        plt.ylabel('precisión')
        plt.plot(self.x_plot, self.y_plot)
        plt.show()
