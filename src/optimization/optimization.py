import numpy as np
from src.configuration.config import Config


class Optimization:
    def __init__(self):
        config = Config()
        self.maximum = config.get_param('algorithm_parameters.maximization')

    def bent_cigar_function(self, x):
        if self.maximum:
            return x[0]**2 + 10**6 * np.sum(xi**2 for xi in x[1:])
        return 1/(x[0]**2 + 10**6 * np.sum(xi**2 for xi in x[1:]))

    def hypersphere(self, x):
        if self.maximum:
            return np.sum(xi**2 for xi in x)
        return 1/np.sum(xi**2 for xi in x)