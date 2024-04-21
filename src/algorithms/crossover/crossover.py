import numpy as np
import random
from src.population.specimen import Specimen
from typing import Tuple


# TODO: Poprawić krosowanie aby działało na genach nie chromosomach
# TODO: Kamil poprawić swoje funkcje
# TODO: Julian wrzucić swoją funkcję krzyżującą


class Crossover:
    def __init__(self, crossover_prob=0.9, cross_method='single_point_crossover', swap_prob=0.5, max=False):
        self.crossover_prob = crossover_prob
        self.cross_method = cross_method
        self.swap_prob = swap_prob
        self.children = []
        self.max = max

    def arithmetic_crossover(self, specimen1, specimen2):
        alpha = np.random.uniform(0, 1)

        child1_chromosomes = []
        child2_chromosomes = []

        for i in range(len(specimen1)):
            chromosome1 = specimen1[i]
            chromosome2 = specimen2[i]

            new_chromosome1 = alpha * chromosome1 + (1 - alpha) * chromosome2
            new_chromosome2 = (1 - alpha) * chromosome1 + alpha * chromosome2

            child1_chromosomes.append(new_chromosome1)
            child2_chromosomes.append(new_chromosome2)

        child1 = Specimen.from_chromosomes(child1_chromosomes, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)
        child2 = Specimen.from_chromosomes(child2_chromosomes, specimen2.boundaries, specimen2.accuracy,
                                           specimen2.fitness_function)
        self.children.append(child1)
        self.children.append(child2)

    def linear_crossover(self, specimen1, specimen2):
        child1_chromosomes = []
        child2_chromosomes = []
        child3_chromosomes = []

        for i in range(len(specimen1)):
            chromosome1 = specimen1[i]
            chromosome2 = specimen2[i]

            new_chromosome1 = 0.5 * chromosome1[i] + 0.5 * chromosome2[i]
            new_chromosome2 = 1.5 * chromosome1 - 0.5 * chromosome2
            new_gene3 = -0.5 * chromosome1 + 1.5 * chromosome2

            child1_chromosomes.append(new_chromosome1)
            child2_chromosomes.append(new_chromosome2)
            child3_chromosomes.append(new_gene3)

        child1 = Specimen.from_chromosomes(child1_chromosomes, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)
        child2 = Specimen.from_chromosomes(child2_chromosomes, specimen2.boundaries, specimen2.accuracy,
                                           specimen2.fitness_function)
        child3 = Specimen.from_chromosomes(child3_chromosomes, specimen2.boundaries, specimen2.accuracy,
                                           specimen2.fitness_function)
        best_children = sorted((child1, child2, child3), key=lambda x: x.get_fitness(), reverse=self.max)[:2]

        self.children.append(best_children[0])
        self.children.append(best_children[1])

    def arithmetic_crossover(self, specimen1, specimen2):
        alpha = np.random.uniform(0, 1)

        child1_chromosomes = []
        child2_chromosomes = []

        for i in range(len(specimen1)):
            chromosome1 = specimen1[i]
            chromosome2 = specimen2[i]

            new_chromosome1 = alpha * chromosome1 + (1 - alpha) * chromosome2
            new_chromosome2 = (1 - alpha) * chromosome1 + alpha * chromosome2

            child1_chromosomes.append(new_chromosome1)
            child2_chromosomes.append(new_chromosome2)

        child1 = Specimen.from_chromosomes(child1_chromosomes, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)
        child2 = Specimen.from_chromosomes(child2_chromosomes, specimen2.boundaries, specimen2.accuracy,
                                           specimen2.fitness_function)
        self.children.append(child1)
        self.children.append(child2)

    def average_crossover(self, specimen1, specimen2):

        child1_chromosomes = []

        for i in range(len(specimen1)):
            chromosome1 = specimen1[i]
            chromosome2 = specimen2[i]

            new_chromosome = (chromosome1 + chromosome2) / 2

            child1_chromosomes.append(new_chromosome)

        child1 = Specimen.from_chromosomes(child1_chromosomes, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)

        self.children.append(child1)

    # used for center of mass crossover
    def blx_crossover(self, specimen1, specimen2):
        child = []

        for i in range(len(specimen1)):
            min_val = min(specimen1[i], specimen2[i])
            max_val = max(specimen1[i], specimen2[i])
            range_val = max_val - min_val
            lower_bound = min_val - 0.2 * range_val
            upper_bound = max_val + 0.2 * range_val
            child.append(random.uniform(lower_bound, upper_bound))

        return child

    def center_of_mass_crossover(self, specimen1, specimen2):
        center_of_mass = (np.sum(specimen1) + np.sum(specimen2)) / 2

        for parent in (specimen1, specimen2):
            temporary_vector = 2 * center_of_mass - parent
            child = self.blx_crossover(temporary_vector, parent)

            self.children.append(Specimen.from_chromosomes(child, specimen1.boundaries, specimen1.accuracy,
                                                           specimen1.fitness_function))

    def imperfect_crossover(
            self,
            parent1: np.ndarray,
            parent2: np.ndarray,
            add_probability: float = 0.3,
            delete_probability: float = 0.6) -> Tuple[np.ndarray, np.ndarray]:

        size_x = len(parent1)
        size_y = len(parent2)
        size = min(size_x, size_y)
        point = random.randint(1, size)
        child1, child2 = np.zeros(size_x), np.zeros(size_y)

        if random.random() < add_probability:
            child1[:point - 1] = parent1[:point - 1]
            child1[point] = np.random.uniform(min(parent1), max(parent1))
            child1[point + 1:] = parent2[point + 1:size_y]

            child2[:point - 1] = parent2[:point - 1]
            child2[point] = np.random.uniform(min(parent2), max(parent2))
            child2[point + 1:] = parent1[point + 1:size_x]
        elif random.random() < delete_probability:
            child1[:point - 1] = parent1[:point - 1]
            child1[point:] = parent2[point:size_y]

            child2[:point - 1] = parent2[:point - 1]
            child2[point:] = parent1[point:size_x]
        else:
            child1[:point] = parent1[:point]
            child1[point:] = parent2[point:]

            child2[:point] = parent2[:point]
            child2[point:] = parent1[point:]

        return child1, child2
