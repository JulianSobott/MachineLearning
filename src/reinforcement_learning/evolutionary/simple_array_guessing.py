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
        self.survived_epochs = 0
        self.num_hits = 0
        self.fitness_this_epoch = 0

    def copy(self):
        c = Agent(self.array_size)
        c.nn = self.nn.copy()
        c.survived_epochs = self.survived_epochs
        return c

    def mutate(self, mutation_rate_gene, mutation_deltas_gene):
        for i in range(self.nn.weights[0].shape[0]):
            if random.random() < mutation_rate_gene:
                # += instead of = maybe
                update_value = random.randint(mutation_deltas_gene[0], mutation_deltas_gene[1])
                mutation_value = self.nn.weights[0][i][0] + update_value
                self.nn.weights[0][i][0] = mutation_value

    def make_move(self, array: np.ndarray):
        calculated_index = self.nn.calc_output(array)[0]
        target_index = array.nonzero()[0][0]
        diff = abs(calculated_index - target_index)
        self.fitness_this_epoch += diff

    def __repr__(self):
        return f"Agent: weights={self.nn.weights[0].reshape(self.array_size)}, epochs={self.survived_epochs}"


class ArrayGuessing(EvolutionaryAlgorithm):

    def __init__(self, array_size: int):
        super().__init__(population_size=10, mutation_rate_chromosome=0.01, mutation_rate_gene=0.3)
        self.array_size = array_size
        self.perfect_agents = []

    def new_agent(self):
        return Agent(self.array_size)

    def run_epoch(self):
        for _ in range(20):
            array = np.zeros(self.array_size)
            array[np.random.randint(0, self.array_size)] = 1
            for agent in self.population:
                agent: Agent
                agent.make_move(array)

        for agent in self.population:
            agent: Agent
            agent.survived_epochs += 1

    def end_epoch(self):
        if self.epoch % 1000 == 0:
            print(f"Epoch: {self.epoch}. avg_fitness={sum(self.fitness_scores)/self.population_size}")
        if self.epoch % 5000 == 0:
            print(self.population)

    def select_parents(self):
        parents = []
        for agent, fitness in zip(self.population, self.fitness_scores):
            if fitness == 0:
                parents.append(agent.copy())
                parents.append(agent.copy())
        parents = parents[:len(self.population)]
        return parents

    # def mutate_population(self, population):
    #     for child in population:
    #         child: Agent
    #         if ((random.random() < 0.6 and child.fitness_this_epoch > 5) or
    #            (random.random() < 0.1 and child.fitness_this_epoch <= 5)):
    #             child.mutate(self.mutation_rate_gene, self.mutation_deltas_gene)
    #     return population

    def evaluate_agent_fitness(self, agent: Agent) -> int:
        fitness = agent.fitness_this_epoch
        if fitness == 0:    # perfect agent
            self.perfect_agents.append(agent.copy())
            # print(f"{agent}")
            # print(f"Hit: perfect_agents={len(self.perfect_agents)}, epoch={self.epoch}")
        return fitness

    def on_finish(self):
        print(f"Finished after {self.epoch} epochs")
        for agent in self.population:
            agent: Agent
            print(f"{agent}")

    def has_run_ended(self):
        return sum(self.fitness_scores)/self.population_size < 10 and self.epoch > 10


if __name__ == '__main__':
    ArrayGuessing(5).run()
