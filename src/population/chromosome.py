import numpy as np
from typing import Tuple
from math import log2, ceil


class Chromosome:
    def __init__(self, boundaries: Tuple[float, float], accuracy: int):
        self.boundaries = boundaries
        self.accuracy = accuracy
        self.chromosome_length = ceil(log2((boundaries[1] - boundaries[0]) * 10 ** accuracy))
        self.chromosome = np.random.uniform(size=self.chromosome_length)

    def get_boundaries(self) -> Tuple[float, float]:
        return self.boundaries

    def get_accuracy(self) -> int:
        return self.accuracy

    def get_chromosome_length(self) -> int:
        return self.chromosome_length

    def get_chromosome(self) -> np.ndarray:
        return self.chromosome

    def set_chromosome(self, chromosome):
        self.chromosome = chromosome

    def decode_uniform_chromosome(self) -> float:
        return sum(value for value in self.chromosome)

    def __str__(self) -> str:
        return f'Chromosome: {self.chromosome}'

