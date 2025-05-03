import numpy as np

class DistanceCalculator:
    def euclidean(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def manhattan(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def minkowski(self, x1, x2, p=3):
        return np.power(np.sum(np.abs(x1 - x2) ** p), 1 / p)
