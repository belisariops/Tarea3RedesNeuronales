import random
from abc import ABC, abstractmethod


class AbstractGeneticAlgorithm(ABC):
    def __init__(self, initial_population):
        self.initial_population = initial_population
        self.mutation_rate = 0
        self.population_size = 0
        self.number_genes = 0
        self.generation = 0
        self.population = []
        self.tournament_selection_individuals = int(self.initial_population*0.2)

    @abstractmethod
    def initialize_population(self, n):
        for index in range(self.initial_population):
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
    def reproduction(self):
        pass

    @abstractmethod
    def run(self):
        pass
