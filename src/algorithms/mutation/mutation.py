import numpy as np


class Mutation:
    def __init__(self, mutation_rate, mutation_method):
        self.mutation_rate = mutation_rate
        self.mutation_method = mutation_method

    def even_mutation(self, specimen, a=1, b=10):

        for i in range(len(specimen.get_specimen())):
            chromosome = specimen.get_specimen()[i].get_chromosome()

            chromosome[np.random.randint(0, len(chromosome) - 1)] = np.random.rand(a, b)

            specimen.get_specimen()[i].set_chromosome(chromosome)

        return specimen

    def gauss_mutation(self, specimen):

        for i in range(len(specimen.get_specimen())):
            chromosome = specimen.get_specimen()[i].get_chromosome()

            for j in range(len(chromosome)):
                chromosome[j] += np.random.normal()

            specimen.get_specimen()[i].set_chromosome(chromosome)

        return specimen
