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
from typing import List, Union, Optional, Tuple

__all__ = ['Chromosome', 'Population', 'FitnessScore', 'EvolutionaryAlgorithm']


class Chromosome(abc.ABC):
    pass


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

    def evaluate_agent_fitness(self, agent: Chromosome):
        raise NotImplementedError

    def create_new_population(self):
        raise NotImplementedError

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
