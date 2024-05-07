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

        for i in range(specimen1.get_number_of_chromosomes()):
            chromosome1 = specimen1.specimen[i].chromosome
            chromosome2 = specimen2.specimen[i].chromosome

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

        for i in range(specimen1.get_number_of_chromosomes()):
            chromosome1 = specimen1.specimen[i].chromosome
            chromosome2 = specimen2.specimen[i].chromosome

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

        for i in range(specimen1.get_number_of_chromosomes()):
            chromosome1 = specimen1.specimen[i].chromosome
            chromosome2 = specimen2.specimen[i].chromosome

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

        for i in range(specimen1.get_number_of_chromosomes()):
            new_chromosome1 = [0] * len(specimen1.specimen[i].chromosome)
            new_chromosome2 = [0] * len(specimen1.specimen[i].chromosome)

            min_val = np.min(np.minimum(specimen1.specimen[i].chromosome, specimen2.specimen[i].chromosome))
            max_val = np.max(np.maximum(specimen1.specimen[i].chromosome, specimen2.specimen[i].chromosome))
            range_val = max_val - min_val
            lower_bound = min_val - 0.2 * range_val
            upper_bound = max_val + 0.2 * range_val

            for j in range(len(specimen1.specimen[i].chromosome)):
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

        for i in range(specimen1.get_number_of_chromosomes()):
            new_chromosome1 = [0] * len(specimen1.specimen[i].chromosome)
            new_chromosome2 = [0] * len(specimen1.specimen[i].chromosome)

            min_val = np.min(np.minimum(specimen1.specimen[i].chromosome, specimen2.specimen[i].chromosome))
            max_val = np.max(np.maximum(specimen1.specimen[i].chromosome, specimen2.specimen[i].chromosome))
            range_val = max_val - min_val
            lower_bound = min_val - 0.2 * range_val
            upper_bound = max_val + 0.3 * range_val

            for j in range(len(specimen1.specimen[i].chromosome)):
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
        temporary_vector1 = [0] * specimen1.get_number_of_chromosomes()
        temporary_vector2 = [0] * specimen1.get_number_of_chromosomes()
        center_of_mass = 0

        for i in range(specimen1.get_number_of_chromosomes()):
            center_of_mass += np.sum(specimen1.specimen[i].chromosome) + np.sum(specimen2.specimen[i].chromosome)

        center_of_mass = center_of_mass / 2

        for j in range(specimen1.get_number_of_chromosomes()):
            temporary_vector1[j] = -1 * specimen1.specimen[j].chromosome + 2 * center_of_mass
            temporary_vector2[j] = -1 * specimen2.specimen[j].chromosome + 2 * center_of_mass

        child1 = Specimen.from_chromosomes(temporary_vector1, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)
        child2 = Specimen.from_chromosomes(temporary_vector2, specimen1.boundaries, specimen1.accuracy,
                                           specimen1.fitness_function)

        self.blend_crossover_alpha(child1, specimen1)
        self.blend_crossover_alpha(child2, specimen2)

    def imperfect_crossover(self, specimen1, specimen2, add_probability: float = 0.3, delete_probability: float = 0.6):

        size_x = specimen1.get_number_of_chromosomes()
        size_y = specimen2.get_number_of_chromosomes()

        size = min(size_x, size_y)
        point = random.randint(1, size)

        child1_chromosomes, child2_chromosomes = [None] * size_x, [None] * size_y

        spec1List = [x.get_chromosome() for x in specimen1.specimen]
        spec2List = [x.get_chromosome() for x in specimen2.specimen]

        if random.random() < add_probability:
            child1_chromosomes[:point - 1] = specimen1.specimen[:point - 1]
            child1_chromosomes[point + 1:] = specimen2.specimen[point + 1:size_y]
            child1_chromosomes[point] = np.random.uniform(np.min(spec1List), np.max(spec1List))

            child2_chromosomes[:point - 1] = specimen2.specimen[:point - 1]
            child2_chromosomes[point + 1:] = specimen1.specimen[point + 1:size_x]
            child2_chromosomes[point] = np.random.uniform(np.min(spec2List), np.max(spec2List))


        elif random.random() < delete_probability:
            child1_chromosomes[:point - 1] = specimen1.specimen[:point - 1]
            child1_chromosomes[point:] = specimen2.specimen[point:size_y]
            child1_chromosomes[point] = np.zeros(len(spec1List[0]))

            child2_chromosomes[:point - 1] = specimen2.specimen[:point - 1]
            child2_chromosomes[point:] = specimen1.specimen[point:size_x]
            child2_chromosomes[point] = np.zeros(len(spec1List[0]))
        else:
            child1_chromosomes[:point] = specimen1.specimen[:point]
            child1_chromosomes[point:] = specimen2.specimen[point:]

            child2_chromosomes[:point] = specimen2.specimen[:point]
            child2_chromosomes[point:] = specimen1.specimen[point:]



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

        for i in range(specimen1.get_number_of_chromosomes()):
            new_chromosome1 = [0] * len(specimen1.specimen[i].chromosome)
            new_chromosome2 = [0] * len(specimen1.specimen[i].chromosome)

            chromosome1 = specimen1.specimen[i].chromosome
            chromosome2 = specimen2.specimen[i].chromosome

            for j in range(len(specimen1.specimen[i].chromosome)):
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

    def cross(self, parent1, parent2):
        self.children = []

        if self.cross_method == 'arithmetic_crossover':
            self.arithmetic_crossover(parent1, parent2)
        elif self.cross_method == 'linear_crossover':
            self.linear_crossover(parent1, parent2)
        elif self.cross_method == 'average_crossover':
            self.average_crossover(parent1, parent2)
        elif self.cross_method == 'blend_crossover_alpha':
            self.blend_crossover_alpha(parent1, parent2)
        elif self.cross_method == 'blend_crossover_beta':
            self.blend_crossover_beta(parent1, parent2)
        elif self.cross_method == 'center_of_mass_crossover':
            self.center_of_mass_crossover(parent1, parent2)
        elif self.cross_method == 'imperfect_crossover':
            self.imperfect_crossover(parent1, parent2)
        elif self.cross_method == 'linear3_crossover':
            self.linear3_crossover(parent1, parent2)

        return self.children
