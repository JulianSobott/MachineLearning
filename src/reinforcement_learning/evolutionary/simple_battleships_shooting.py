"""
Very simplified example, to test kinda battleships shooting.

Given a small field. represented by an 2D array.

e.g.

::
    -----
    |1|0|
    -----
    |0|0|
    -----

array = [[1, 0], [0, 0]]    # input

The goal of this program is to output the index, where the 1 is.
In this case expected output would be: [0, 0]

Algorithm
---------

Using the evolutionary algorithm with NeuralNetworks as agents

Network
-------

num_input: width * height
output:
    - [x, y]
    - [index] (map array to 1D array)
    - [i0, i1, i2, i3] (map each output to one index)

"""
import random
from typing import Tuple, List
import numpy as np

from MLlib.algorithms.evolutionary import EvolutionaryAlgorithm, Chromosome
from MLlib.NeuralNetworks import SimpleNeuralNetwork, activation_functions as af


class Agent(Chromosome):

    def __init__(self, playground_size, playground):
        self.nn = SimpleNeuralNetwork(layers=[playground_size * playground_size, 5, 2],
                                      activation_functions=(af.linear, af.linear),
                                      weights_min_max=(-playground_size, playground_size)
                                      )
        self.playground_size = playground_size
        self.playground = np.array(playground)

        self.shoot_position = None

    def shoot(self):
        in_values = self.playground.reshape(self.playground_size * self.playground_size)
        out_values = self.nn.calc_output(in_values)
        self.shoot_position = out_values

    def copy(self):
        c = Agent(self.playground_size, self.playground[:])
        c.nn = self.nn.copy()
        return c

    def mutate(self, mutation_rate_gene, mutation_deltas_gene):
        for i in range(self.nn.weights[0].shape[0]):
            if random.random() < mutation_rate_gene:
                # += instead of = maybe
                update_value = random.randint(mutation_deltas_gene[0], mutation_deltas_gene[1])
                mutation_value = self.nn.weights[0][i][0] + update_value
                self.nn.weights[0][i][0] = mutation_value


class ShootingSystem(EvolutionaryAlgorithm):

    def __init__(self, playground: List[List[int]]):
        super().__init__()
        self.playground = playground
        self.playground_size = len(playground)

        self.num_hits = 0

    def new_agent(self):
        return Agent(self.playground_size, self.playground)

    def run_epoch(self):
        for agent in self.population:
            agent: Agent
            agent.shoot()

    def end_epoch(self):
        if self.epoch % 1000 == 0:
            avg_fitness = sum(self.fitness_scores) / len(self.population)
            print(f"{self.epoch}: avg_fitness={avg_fitness}, hits={self.num_hits}")

    def evaluate_agent_fitness(self, agent: Agent):
        x = agent.shoot_position[0]
        y = agent.shoot_position[1]
        if not 0 < x < self.playground_size or not 0 < y < self.playground_size:
            return 0
        if agent.playground[x][y] == 1:
            self.num_hits += 1
            return 5
        else:
            return 1

    def on_finish(self):
        pass

    def has_run_ended(self):
        return self.num_hits > 10


if __name__ == '__main__':
    playground = [[0, 0], [0, 1]]
    s = ShootingSystem(playground)
    s.run()
