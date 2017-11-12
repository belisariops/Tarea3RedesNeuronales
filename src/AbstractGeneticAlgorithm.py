import random
from abc import ABC, abstractmethod

from src.SerializedNetwork import SerializeNetwork


class AbstractGeneticAlgorithm(ABC):
    def __init__(self, initial_population):
        self.mutation_rate = 0.7
        self.population_size = initial_population
        self.number_genes = 0
        self.generation = 0
        self.population = []
        self.tournament_selection_individuals = int(self.population_size * 1.0)
        self.condition = lambda x: x

    @abstractmethod
    def initialize_population(self, n):
        for index in range(self.population_size):
            individual = SerializeNetwork()
            for num_neuerons in range(self.number_genes):
                individual.serialized_values.append(random.uniform(-3.0, 3.0))
            self.population.append(individual)

    @abstractmethod
    def evaluate_fitness(self, individual):
        pass

    def selection(self):
        best_individual = None
        index_best_individual = -1
        for i in range(self.tournament_selection_individuals):
            index = random.randint(0, len(self.population) - 1)
            individual = self.population[index]
            if (best_individual is None) or (
                        self.evaluate_fitness(individual) > self.evaluate_fitness(best_individual)):
                best_individual = individual
                index_best_individual = index
        return best_individual

    def cross_over(self, father, mother):
        cross_over_point = random.randint(0, self.number_genes)
        first_child = SerializeNetwork()
        second_child = SerializeNetwork()
        first_child.serialized_values = father.serialized_values[0:cross_over_point] + mother.serialized_values[cross_over_point:]
        second_child.serialized_values = mother.serialized_values[0:cross_over_point] + father.serialized_values[cross_over_point:]
        return first_child, second_child

    @abstractmethod
    def run(self):
        pass

    def mutate(self, individual):
        pass
