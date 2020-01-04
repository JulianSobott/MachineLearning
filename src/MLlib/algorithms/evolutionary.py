"""
:module: MLlib.algorithms.evolutionary
:synopsis: abstract class for Evolutionary algorithms
:author: Julian Sobott

public classes
---------------

.. autoclass:: EvolutionaryAlgorithm
    :members:
.. autoclass:: Chromosome
    :members:
"""
import abc
import random

import numpy as np
from typing import List, Union, Optional, Tuple

__all__ = ['Chromosome', 'Population', 'FitnessScore', 'EvolutionaryAlgorithm']


class Chromosome(abc.ABC):

    def copy(self):
        raise NotImplementedError

    def mutate(self, mutation_rate_gene, mutation_deltas_gene):
        raise NotImplementedError


Population = List[Chromosome]
FitnessScore = Union[int, float]


class EvolutionaryAlgorithm(abc.ABC):

    def __init__(self,
                 population_size: int = 10,
                 seed: Optional[int] = None,
                 mutation_rate_chromosome: float = 0.3,
                 mutation_rate_gene: float = 0.3,
                 mutation_deltas_gene: Tuple[Union[float, int], Union[float, int]] = (1, 1),
                 ):
        # metadata
        self.population_size: int = population_size
        self.seed: Optional[int] = seed
        self.mutation_rate_chromosome: float = mutation_rate_chromosome
        self.mutation_rate_gene: float = mutation_rate_gene
        self.mutation_deltas_gene = mutation_deltas_gene

        # working data
        self.population: Population = []
        self.fitness_scores = []
        self.epoch = 0

    def init_population(self):
        self.population = [self.new_agent() for _ in range(self.population_size)]

    def new_agent(self):
        raise NotImplementedError

    def run_epoch(self):
        raise NotImplementedError

    def end_epoch(self):
        """Epoch has ended. Fitness_scores are updated. No new population is created"""
        raise NotImplementedError

    def _end_epoch(self):
        self.evaluate()
        self.end_epoch()
        self.epoch += 1

    def evaluate(self):
        for agent in self.population:
            self.fitness_scores.append(self.evaluate_agent_fitness(agent))

    def evaluate_agent_fitness(self, agent: Chromosome) -> int:
        raise NotImplementedError

    def create_new_population(self):
        selected_parents = self.select_parents()
        new_population = self.crossover(selected_parents)
        new_population = self.mutate_population(new_population)
        new_population += [self.new_agent() for _ in range(0, len(self.population) - len(new_population))]
        self.population = new_population

    def select_parents(self) -> Population:
        parent_population_ratio = 3/4
        sum_fitness = sum(self.fitness_scores)
        np_fs = np.array(self.fitness_scores)
        inverted_fitness_scores = sum_fitness - np_fs
        sum_inverted_fs = np.sum(inverted_fitness_scores)
        if sum_inverted_fs == 0:
            indices = np.random.choice(len(self.fitness_scores), len(self.fitness_scores))
        else:
            weights = inverted_fitness_scores / sum_inverted_fs
            indices = np.random.choice(len(self.fitness_scores),
                                       int(parent_population_ratio * len(self.fitness_scores)),
                                       p=weights)

        return [self.population[i].copy() for i in indices]

    def crossover(self, parents: Population):
        return parents  # TODO

    def mutate_population(self, population: Population):
        for child in population:
            if random.random() < self.mutation_rate_chromosome:
                child.mutate(self.mutation_rate_gene, self.mutation_deltas_gene)
        return population

    def run(self):
        self.init_population()
        while not self.has_run_ended():
            self.run_epoch()
            self._end_epoch()
            self.create_new_population()
            self.fitness_scores = []
        self.on_finish()

    def on_finish(self):
        raise NotImplementedError

    def has_run_ended(self):
        raise NotImplementedError
