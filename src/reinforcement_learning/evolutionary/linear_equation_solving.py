"""
http://www.worldscientificnews.com/wp-content/uploads/2015/07/WSN-19-2015-148-167.pdf

"""
from typing import List
import random
from string import ascii_lowercase
import numpy as np

Gene = int
Chromosome = List[Gene]
Population = List[Chromosome]
FitnessScore = int

MIN_GENE_VALUE = 0
MAX_GENE_VALUE = 40
MUTATION_RATE = 0.01


class Equation:

    def __init__(self, target: int, coefficients: tuple):
        self.target = target
        self.coefficients = coefficients

    def calc_diff(self, values: Chromosome):
        solution = -self.target
        for co, value in zip(self.coefficients, values):
            solution += co * value
        return solution

    def num_coefficients(self) -> int:
        return len(self.coefficients)

    def __repr__(self):
        s = f"{self.target} = "
        for i, c in enumerate(self.coefficients):
            s += f"{c}*{ascii_lowercase[i]} + "
        s = s[:-2]
        return s


def init_population(size: int, num_genes: int) -> Population:
    return [random_chromosome(num_genes) for _ in range(size)]


def random_chromosome(num_genes: int):
    return [random.randint(MIN_GENE_VALUE, MAX_GENE_VALUE) for _ in range(num_genes)]


def evaluate(population: Population, equation: Equation) -> List[FitnessScore]:
    fitness_scores = []
    for chrome in population:
        fitness = abs(equation.calc_diff(chrome))
        fitness_scores.append(fitness)
    return fitness_scores


def create_new_population(fitness_scores: List[FitnessScore], population: Population) -> Population:
    # SELECT
    sum_fitness = sum(fitness_scores)
    weights = [score/sum_fitness for score in fitness_scores]
    indices = np.random.choice(len(fitness_scores), len(fitness_scores), p=weights)
    # len(parents) = len(population)
    parents = [population[i] for i in indices]

    # CROSSOVER
    new_population = []
    for i in range(0, len(parents), 2):
        p1 = parents[i]
        p2 = parents[i + 1]
        child = p1[:int(len(p1)/2)] + p2[int(len(p2)/2):]
        new_population.append(child)

    # MUTATION
    for child in new_population:
        for i in range(len(child)):
            if random.random() < MUTATION_RATE:
                child[i] = random.randint(MIN_GENE_VALUE, MAX_GENE_VALUE)

    # FILL to original size
    new_population += [random_chromosome(len(child)) for _ in range(0, len(population) - len(new_population))]
    return new_population


def solve_equation(equation: Equation):
    population_size = 10
    population = init_population(population_size, equation.num_coefficients())
    fitness_scores: list = evaluate(population, equation)
    avg_fitness = sum(fitness_scores) / len(population)
    generation = 0
    solutions = []
    while len(solutions) < 10:
        population = create_new_population(fitness_scores, population)
        fitness_scores = evaluate(population, equation)
        avg_fitness = sum(fitness_scores)/len(population)
        if generation % 1000 == 0:
            print(f"{generation}: avg_fitness={avg_fitness}, num_solutions={len(solutions)}")
        if 0 in fitness_scores:
            fitness_scores_np = np.array(fitness_scores)
            indices = np.where(fitness_scores_np == 0)[0]
            s = np.array(population)[indices]
            if len(solutions) == 0:
                solutions = s
            else:
                solutions = np.append(solutions, s, axis=0)
        generation += 1
    print(f"FINISHED in generation: {generation}")
    print(f"{generation}: avg fitness: {avg_fitness}")
    print(f"Possible solutions for equation: {equation}")
    print(solutions)
    return solutions


if __name__ == '__main__':
    eq = Equation(57, (1, -2, 3, 1, 2, 7, 3))
    possible_solutions = solve_equation(eq)
