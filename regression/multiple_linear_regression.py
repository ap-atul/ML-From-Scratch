import numpy as np
import copy

class MultipleLinearRegression:
    def __init__(self):
        self._coeff, self._intercept = None, None

    def fit(self, x, y):
        betas = self._estimate(self._transform(x), y)
        self._intercept, self._coeff = betas[0], betas[1: ]

    def predict(self, x): # y = b0 + b1 * x + .. + bi * xi
        return [(self._intercept + sum(np.multiply(row, self._coeff))) for row in x]

    def _transform(self, x):
        return x.insert(0, 'ones', np.ones((x.shape[0], 1)))

    def _estimate(self, x, y): # β = (X^T X)^-1 X^T y
        inversed = np.linalg.inv(x.T.dot(x))
        return inversed.dot(x.T).dot(y)
