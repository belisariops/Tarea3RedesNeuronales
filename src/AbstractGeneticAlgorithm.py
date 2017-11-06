from abc import ABC, abstractmethod


class AbstractGeneticAlgorithm(ABC):
    def __init__(self):
        self.mutation_rate = 0
        self.population_size = 0
        self.number_genes = 0

    @abstractmethod
    def initialize_population(self, n):
        pass

    @abstractmethod
    def evaluate_fitness(self, individual):
        pass

    @abstractmethod
    def selection(self):
        pass

    @abstractmethod
    def reproduction(self):
        pass


    @abstractmethod
    def run(self):
        pass