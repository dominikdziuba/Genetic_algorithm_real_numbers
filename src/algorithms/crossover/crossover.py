import numpy as np
import random
from src.population.specimen import Specimen


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

            new_chromosome1 = [0] * len(chromosome1)
            new_chromosome2 = [0] * len(chromosome2)

            for j in range(len(chromosome1)):
                new_chromosome1[j] = alpha * chromosome1[j] + (1 - alpha) * chromosome2[j]
                new_chromosome2[j] = (1 - alpha) * chromosome1[j] + alpha * chromosome2[j]

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

            new_chromosome1 = [0] * len(chromosome1)
            new_chromosome2 = [0] * len(chromosome2)
            new_chromosome3 = [0] * len(chromosome2)

            for j in range(len(chromosome1)):
                new_chromosome1[j] = 0.5 * chromosome1[j] + 0.5 * chromosome2[j]
                new_chromosome2[j] = 1.5 * chromosome1[j] - 0.5 * chromosome2[j]
                new_chromosome3[j] = -0.5 * chromosome1[j] + 1.5 * chromosome2[j]

            child1_chromosomes.append(new_chromosome1)
            child2_chromosomes.append(new_chromosome2)
            child3_chromosomes.append(new_chromosome3)

        child1 = Specimen.from_chromosomes(child1_chromosomes, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)
        child2 = Specimen.from_chromosomes(child2_chromosomes, specimen2.boundaries, specimen2.accuracy,
                                           specimen2.fitness_function)
        child3 = Specimen.from_chromosomes(child3_chromosomes, specimen2.boundaries, specimen2.accuracy,
                                           specimen2.fitness_function)

        best_children = sorted((child1, child2, child3), key=lambda x: x.get_fitness(), reverse=self.max)[:2]

        self.children.append(best_children[0])
        self.children.append(best_children[1])

    def average_crossover(self, specimen1, specimen2):

        child1_chromosomes = []

        for i in range(len(specimen1)):
            chromosome1 = specimen1[i]
            chromosome2 = specimen2[i]

            new_chromosome = [0] * len(chromosome1)

            for j in range(len(chromosome1)):
                new_chromosome[j] = (chromosome1[j] + chromosome2[j]) / 2

            child1_chromosomes.append(new_chromosome)

        child1 = Specimen.from_chromosomes(child1_chromosomes, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)

        self.children.append(child1)
        self.children.append(child1)

    def blend_crossover_alpha(self, specimen1, specimen2):
        child1_chromosomes = []
        child2_chromosomes = []

        for i in range(len(specimen1)):
            new_chromosome1 = [0] * len(specimen1[i])
            new_chromosome2 = [0] * len(specimen1[i])

            min_val = min(specimen1[i], specimen2[i])
            max_val = max(specimen1[i], specimen2[i])
            range_val = max_val - min_val
            lower_bound = min_val - 0.2 * range_val
            upper_bound = max_val + 0.2 * range_val

            for j in range(len(specimen1[i])):
                new_chromosome1[j] = random.uniform(lower_bound, upper_bound)
                new_chromosome2[j] = random.uniform(lower_bound, upper_bound)

            child1_chromosomes.append(new_chromosome1)
            child2_chromosomes.append(new_chromosome2)

        child1 = Specimen.from_chromosomes(child1_chromosomes, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)
        child2 = Specimen.from_chromosomes(child2_chromosomes, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)
        self.children.append(child1)
        self.children.append(child2)

    def blend_crossover_beta(self, specimen1, specimen2):
        child1_chromosomes = []
        child2_chromosomes = []

        for i in range(len(specimen1)):
            new_chromosome1 = [0] * len(specimen1[i])
            new_chromosome2 = [0] * len(specimen1[i])

            min_val = min(specimen1[i], specimen2[i])
            max_val = max(specimen1[i], specimen2[i])
            range_val = max_val - min_val
            lower_bound = min_val - 0.2 * range_val
            upper_bound = max_val + 0.3 * range_val

            for j in range(len(specimen1[i])):
                new_chromosome1[j] = random.uniform(lower_bound, upper_bound)
                new_chromosome2[j] = random.uniform(lower_bound, upper_bound)

            child1_chromosomes.append(new_chromosome1)
            child2_chromosomes.append(new_chromosome2)

        child1 = Specimen.from_chromosomes(child1_chromosomes, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)
        child2 = Specimen.from_chromosomes(child2_chromosomes, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)
        self.children.append(child1)
        self.children.append(child2)

    def center_of_mass_crossover(self, specimen1, specimen2):
        temporary_vector1 = []
        temporary_vector2 = []

        center_of_mass = (np.sum(specimen1) + np.sum(specimen2)) / 2

        for j in range(len(specimen1)):
            temporary_vector1[j] = -1 * specimen1[j] + 2 * center_of_mass
            temporary_vector2[j] = -1 * specimen2[j] + 2 * center_of_mass

        child1 = Specimen.from_chromosomes(temporary_vector1, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)
        child2 = Specimen.from_chromosomes(temporary_vector2, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)

        self.blend_crossover_alpha(child1, specimen1)
        self.blend_crossover_alpha(child2, specimen2)

    def imperfect_crossover(self, specimen1, specimen2, add_probability: float = 0.3, delete_probability: float = 0.6):

        size_x = len(specimen1)
        size_y = len(specimen2)
        size = min(size_x, size_y)
        point = random.randint(1, size)
        child1_chromosomes, child2_chromosomes = np.zeros(size_x), np.zeros(size_y)

        if random.random() < add_probability:
            child1_chromosomes[:point - 1] = specimen1[:point - 1]
            child1_chromosomes[point] = np.random.uniform(min(specimen1), max(specimen1))
            child1_chromosomes[point + 1:] = specimen2[point + 1:size_y]

            child2_chromosomes[:point - 1] = specimen2[:point - 1]
            child2_chromosomes[point] = np.random.uniform(min(specimen2), max(specimen2))
            child2_chromosomes[point + 1:] = specimen1[point + 1:size_x]
        elif random.random() < delete_probability:
            child1_chromosomes[:point - 1] = specimen1[:point - 1]
            child1_chromosomes[point:] = specimen2[point:size_y]

            child2_chromosomes[:point - 1] = specimen2[:point - 1]
            child2_chromosomes[point:] = specimen1[point:size_x]
        else:
            child1_chromosomes[:point] = specimen1[:point]
            child1_chromosomes[point:] = specimen2[point:]

            child2_chromosomes[:point] = specimen2[:point]
            child2_chromosomes[point:] = specimen1[point:]

        child1 = Specimen.from_chromosomes(child1_chromosomes, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)
        child2 = Specimen.from_chromosomes(child2_chromosomes, specimen2.boundaries, specimen2.accuracy,
                                           specimen2.fitness_function)

        self.children.append(child1)
        self.children.append(child2)

    def linear3_crossover(self, specimen1, specimen2, alpha=random.random()):

        if alpha >= 0.5:
            beta = (2 * alpha) ** (1 / (0.5 + 1))
        else:
            beta = (1 / (2 * (1 - alpha))) ** (1 / (0.5 + 1))

        child1_chromosomes = []
        child2_chromosomes = []

        for i in range(len(specimen1)):
            new_chromosome1 = [0] * len(specimen1[i])
            new_chromosome2 = [0] * len(specimen1[i])

            chromosome1 = specimen1[i]
            chromosome2 = specimen2[i]

            for j in range(len(specimen1[i])):
                new_chromosome1[j] = 0.5 * ((1 + beta) * chromosome1[j] + (1 - beta) * chromosome2[j])
                new_chromosome2[j] = 0.5 * ((1 - beta) * chromosome1[j] + (1 + beta) * chromosome2[j])

            child1_chromosomes.append(new_chromosome1)
            child2_chromosomes.append(new_chromosome2)

        child1 = Specimen.from_chromosomes(child1_chromosomes, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)
        child2 = Specimen.from_chromosomes(child2_chromosomes, specimen2.boundaries, specimen2.accuracy,
                                           specimen2.fitness_function)

        self.children.append(child1)
        self.children.append(child2)
