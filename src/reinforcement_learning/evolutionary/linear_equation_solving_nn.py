"""
Solving linear equations with the evolutionary algorithm using a neural network

Chromosome := NeuralNetwork

Equation
--------

y = Q1*a + Q2*b + Q3*c ...
y := solution
Qn := coefficients (predefined constants)
a, b, ... := variables (need to be calculated by NN)

y and Qn values or normalized to map to [0, 1].
-> To get the correct values for the variables, they need to be mapped back.

Network:
--------

num_input: num_coefficients/num_variables
num_output: 1

input_values: coefficients
**weights: variables**
output: solution

activation function out: linear
"""
from typing import List
import random
import numpy as np
from string import ascii_lowercase
from matplotlib import pyplot as plt


from MLlib import SimpleNeuralNetwork, activation_functions as af

Chromosome = SimpleNeuralNetwork
Population = List[Chromosome]
FitnessScore = int

MIN_GENE_VALUE = 0
MAX_GENE_VALUE = 40
MUTATION_RATE_CHROMOSOME = 0.3
MUTATION_RATE_GENE = 0.5
MUTATION_DELTA_GENE = 0.5

ACCEPTING_FITNESS_EPSILON = 0.01

random.seed(100)
#
plt_weights_history_x = [[] for _ in range(10)]
plt_weights_history_y = [[] for _ in range(10)]


class Equation:

    def __init__(self, target: int, coefficients: tuple):
        self.target = target
        self.coefficients = coefficients
        self._np_coefficients = np.array(coefficients)
        self._norm_factor = max(self.target, *self.coefficients)
        self._norm_target = self.target / self._norm_factor

    def calc_diff(self, network: Chromosome):
        predicted_solution = network.calc_output(self._np_coefficients)[0]
        return abs(predicted_solution - self._norm_target)

    def num_coefficients(self) -> int:
        return len(self.coefficients)

    def repr_possible_solution(self, network: SimpleNeuralNetwork) -> str:
        s = f"{self.target} = "
        actual_solution = 0
        for i, c in enumerate(self.coefficients):
            variable = network.weights[0][i][0] * self._norm_factor
            s += f"{c}*{int(variable)} + "
            actual_solution += c * variable
        s = s[:-2] + f" :~ {int(actual_solution)}, fitness: {self.calc_diff(network)}"
        return s

    def __repr__(self):
        s = f"{self.target} = "
        for i, c in enumerate(self.coefficients):
            s += f"{c}*{ascii_lowercase[i]} + "
        s = s[:-2]
        return s


def init_population(size: int, num_genes: int) -> Population:
    return [random_chromosome(num_genes) for _ in range(size)]


def random_chromosome(num_genes: int):
    return SimpleNeuralNetwork([num_genes, 1], [af.linear])


def evaluate(population: Population, equation: Equation) -> List[FitnessScore]:
    fitness_scores = []
    for chrome in population:
        fitness = equation.calc_diff(chrome)
        fitness_scores.append(fitness)
    return fitness_scores


def create_new_population(fitness_scores: List[FitnessScore], population: Population, num_genes: int) -> Population:
    # SELECT
    # Select 'random' chromosomes. better chromosomes have higher chance
    sum_fitness = sum(fitness_scores)
    np_fs = np.array(fitness_scores)
    inverted_fitness_scores = sum_fitness - np_fs
    sum_inverted_fs = np.sum(inverted_fitness_scores)
    weights = inverted_fitness_scores / sum_inverted_fs
    indices = np.random.choice(len(fitness_scores), len(fitness_scores), p=weights)
    # len(parents) = len(population)
    parents = [population[i].copy() for i in indices]

    # # CROSSOVER
    # new_population = []
    # for i in range(0, len(parents), 2):
    #     p1 = parents[i]
    #     p2 = parents[i + 1]
    #     child = p1[:int(len(p1)/2)] + p2[int(len(p2)/2):]
    #     new_population.append(child)
    new_population = parents

    # MUTATION
    for child in new_population:
        if random.random() < MUTATION_RATE_CHROMOSOME:
            mutate_nn(child)

    # FILL to original size
    new_population += [random_chromosome(num_genes) for _ in range(0, len(population) - len(new_population))]
    return new_population


def mutate_nn(nn: Chromosome) -> None:
    for i in range(nn.weights[0].shape[0]):
        if random.random() < MUTATION_RATE_GENE:
            # += instead of = maybe
            mutation_value = min(1, max(0, nn.weights[0][i][0] + random.uniform(-MUTATION_DELTA_GENE,
                                                                                MUTATION_DELTA_GENE)))
            nn.weights[0][i][0] = mutation_value


def solve_equation(equation: Equation):
    population_size = 10
    population = init_population(population_size, equation.num_coefficients())
    fitness_scores: list = evaluate(population, equation)
    avg_fitness = sum(fitness_scores) / len(population)
    generation = 0
    solutions = []
    while len(solutions) < 10:
        population = create_new_population(fitness_scores, population, equation.num_coefficients())
        fitness_scores = evaluate(population, equation)
        avg_fitness = sum(fitness_scores)/len(population)

        # debug plt
        if generation < 16:
            for i, p in enumerate(population):
                plt_weights_history_x[i].append(len(plt_weights_history_x[i]))
                plt_weights_history_y[i].append(p.weights[0][1][0])
        if generation % 1000 == 0:
            print(f"{generation}: avg_fitness={avg_fitness}, num_solutions={len(solutions)}")
            print(f"Current solution: {equation.repr_possible_solution(population[0])}")
        fitness_scores_np = np.array(fitness_scores)
        x = np.where(fitness_scores_np < ACCEPTING_FITNESS_EPSILON)[0]
        if any(np.where(fitness_scores_np < ACCEPTING_FITNESS_EPSILON)[0]):
            indices = np.where(fitness_scores_np < ACCEPTING_FITNESS_EPSILON)[0]
            s = np.array(population)[indices]
            if len(solutions) == 0:
                solutions = s
            else:
                solutions = np.append(solutions, s, axis=0)
            print("added solutions " + str(len(s)))
        generation += 1
    print(f"FINISHED in generation: {generation}")
    print(f"{generation}: avg fitness: {avg_fitness}")
    print(f"Possible solutions for equation: {equation}")
    for solution in solutions:
        print(f"{solution.weights[0].reshape(2) * equation._norm_factor}")

    # plt
    for weight_x, weight_y in zip(plt_weights_history_x, plt_weights_history_y):
        plt.plot(weight_x, weight_y)
    plt.show()
    return solutions


if __name__ == '__main__':
    eq = Equation(10, (2, 2))
    possible_solutions = solve_equation(eq)
