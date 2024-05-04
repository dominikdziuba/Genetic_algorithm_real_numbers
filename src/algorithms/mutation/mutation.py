import numpy as np


class Mutation:
    def __init__(self, mutation_rate, mutation_method):
        self.mutation_rate = mutation_rate
        self.mutation_method = mutation_method

    def even_mutation(self, specimen, a=1, b=10):

        for i in range(len(specimen.get_specimen())):
            chromosome = specimen.get_specimen()[i].get_chromosome()

            chromosome_copy = chromosome.copy()

            chromosome_copy[np.random.randint(0, len(chromosome_copy) - 1)] = np.random.uniform(a, b)

            specimen.get_specimen()[i].set_chromosome(chromosome_copy)

        return specimen

    def gauss_mutation(self, specimen):

        for i in range(len(specimen.get_specimen())):
            chromosome = specimen.get_specimen()[i].get_chromosome()

            chromosome_copy = chromosome.copy()

            for j in range(len(chromosome)):
                chromosome_copy[j] += np.random.normal()

            specimen.get_specimen()[i].set_chromosome(chromosome_copy)

        return specimen

    def mutate(self, specimen):
        if self.mutation_method == 'even_mutation':
            return self.even_mutation(specimen)
        elif self.mutation_method == 'gauss_mutation':
            return self.gauss_mutation(specimen)
