import random
from abc import ABC, abstractmethod


class AbstractGeneticAlgorithm(ABC):
    def __init__(self, initial_population):
        self.mutation_rate = 0
        self.population_size = initial_population
        self.number_genes = 0
        self.generation = 0
        self.population = []
        self.tournament_selection_individuals = int(self.population_size * 0.2)
        self.condition = lambda x: x

    @abstractmethod
    def initialize_population(self, n):
        for index in range(self.population_size):
            individual = []
            for num_neuerons in range(self.number_genes):
                individual.append(random.uniform(0.0, 3.0))

    @abstractmethod
    def evaluate_fitness(self, individual):
        pass

    @abstractmethod
    def selection(self):
        best_individual = None
        for i in range(self.tournament_selection_individuals):
            individual = self.population.pop(random.randint(0, len(self.population) - 1))
            if (best_individual is None) or (
                        self.evaluate_fitness(individual) > self.evaluate_fitness(best_individual)):
                best_individual = individual
        return best_individual

    @abstractmethod
    def cross_over(self, father, mother):
        # TODO Do crossover
        return father, mother

    @abstractmethod
    def run(self):
        pass

    def mutate(self, individual):
        # TODO Mutate individual
        return individual
