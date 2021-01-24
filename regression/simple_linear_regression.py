import numpy as np

def rmse(y_true, y_pred):
    return np.sqrt(np.mean(np.power(y_true - y_pred, 2)))

def mean(values):
    return sum(values) / float(len(values))

def variance(values, mean): # sum( (x - mean(x))^2 )
    return sum([(x - mean)** 2 for x in values])

def covariance(x, y, x_mean, y_mean): # sum((x(i) - mean(x)) * (y(i) - mean(y)))
    return sum([(x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x))])

class LinearRegression: # simple linear regression
    def __init__(self):
        self._m, self._b = None, None

    def fit(self, X, y):
        self._m, self._b = self._coefficients(X, y)

    def predict(self, X):
        return [(self._b + self._m * row) for row in X]

    def _coefficients(self, x, y):
        x_mean, y_mean = mean(x), mean(y)
        m = covariance(x, y, x_mean, y_mean) / variance(x, x_mean)
        b = y_mean - m * x_mean
        return m, b
