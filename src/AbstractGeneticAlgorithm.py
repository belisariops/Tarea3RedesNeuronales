import random
from abc import ABC, abstractmethod


class AbstractGeneticAlgorithm(ABC):
    def __init__(self, initial_population):
        self.mutation_rate = 0.3
        self.population_size = initial_population
        self.number_genes = 0
        self.generation = 0
        self.population = []
        self.tournament_selection_individuals = int(self.population_size * 0.4)
        self.condition = lambda x: x

    @abstractmethod
    def initialize_population(self, n):
        for index in range(self.population_size):
            individual = []
            for num_neuerons in range(self.number_genes):
                individual.append(random.uniform(-100.0, 100.0))
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
        first_child = father[0:cross_over_point] + mother[cross_over_point:]
        second_child = mother[0:cross_over_point] + father[cross_over_point:]
        return first_child, second_child

    @abstractmethod
    def run(self):
        pass

    def mutate(self, individual):
        pass
        # for i in range(len(individual)):
        #     if 0.2 > random.uniform(0, 1):
        #         individual[i] = random.uniform(-100.0, 100.0)
