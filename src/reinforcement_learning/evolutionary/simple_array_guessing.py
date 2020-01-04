"""
Try to output the index of an array, where the 1 is.

Given an array filled with zeros (0) and one one (1).

input: [0, 0, 1, 0]
wanted_output: [2]

"""
import random
import numpy as np

from MLlib.algorithms.evolutionary import EvolutionaryAlgorithm, Chromosome
from MLlib.NeuralNetworks import SimpleNeuralNetwork, activation_functions as af


class Agent(Chromosome):

    def __init__(self, array_size: int):
        self.array_size = array_size
        self.nn = SimpleNeuralNetwork(layers=[array_size, 1], activation_functions=[af.linear],
                                      weights_min_max=(0, array_size))
        self.last_out: np.ndarray = np.array([-1])
        self.current_array = np.zeros(array_size)
        self.survived_epochs = 0
        self.num_hits = 0

    def copy(self):
        c = Agent(self.array_size)
        c.nn = self.nn.copy()
        return c

    def mutate(self, mutation_rate_gene, mutation_deltas_gene):
        for i in range(self.nn.weights[0].shape[0]):
            if random.random() < mutation_rate_gene:
                # += instead of = maybe
                update_value = random.randint(mutation_deltas_gene[0], mutation_deltas_gene[1])
                mutation_value = self.nn.weights[0][i][0] + update_value
                self.nn.weights[0][i][0] = mutation_value

    def make_move(self, array: np.ndarray):
        self.current_array = array
        self.last_out = self.nn.calc_output(array)


class ArrayGuessing(EvolutionaryAlgorithm):

    def __init__(self, array_size: int):
        super().__init__()
        self.array_size = array_size
        self.num_hits = 0

    def new_agent(self):
        return Agent(self.array_size)

    def run_epoch(self):
        array = np.zeros(self.array_size)
        array[np.random.randint(0, self.array_size)] = 1
        for agent in self.population:
            agent: Agent
            agent.make_move(array)
            agent.survived_epochs += 1

    def end_epoch(self):
        if self.epoch % 1000 == 0:
            print(f"Epoch: {self.epoch}. avg_fitness={sum(self.fitness_scores)/self.population_size}")

    def select_parents(self):
        parents = []
        for agent, fitness in zip(self.population, self.fitness_scores):
            if fitness == 0:
                parents.append(agent.copy())
                parents.append(agent.copy())
            elif fitness == 1:
                parents.append(agent.copy())
        parents = parents[:len(self.population)]
        return parents

    def evaluate_agent_fitness(self, agent: Agent) -> int:
        index = agent.last_out[0]
        target_index = agent.current_array.nonzero()[0][0]
        diff = abs(index - target_index)
        if diff == 0:
            self.num_hits += 1
            agent.num_hits += 1
            print(f"Agent: weights={agent.nn.weights[0].reshape(self.array_size)}, array={agent.current_array}, "
                  f"epochs={agent.survived_epochs}")
            print(f"Hit: num_hits={self.num_hits}, epoch={self.epoch}")
        return diff

    def on_finish(self):
        print("Finished")
        for agent in self.population:
            agent: Agent
            print(f"Agent: survived_epochs={agent.survived_epochs}, hits={agent.num_hits}")

    def has_run_ended(self):
        return self.num_hits >= 500


if __name__ == '__main__':
    ArrayGuessing(5).run()
