"""Utility functions """

import numpy as np


def confusion_matrix(true, pred):
    """ Computes a confusion matrix """
    K = len(np.unique(true))  # Number of classes
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result


def train_test_split(X, y, test_size):
    """ Training and testing samples distribution """
    arr_rand = np.random.rand(X.shape[0])
    split = arr_rand < np.percentile(arr_rand, test_size)

    X_train = X[split]
    y_train = y[split]
    X_test = X[~split]
    y_test = y[~split]

    return X_train, X_test, y_train, y_test


def accuracy_score(y_true, y_predictions):
    """ Calculates the accuracy score for the model"""
    correct = 0
    for yt, yp in zip(y_true, y_predictions):
        if yt == yp:
            correct += 1

    return correct / len(y_predictions)


def mean_squared_error(y_true, y_pred):
    """ Returns the mean squared error between y_true and y_pred """
    mse = np.mean(np.power(y_true - y_pred, 2))
    return mse


def root_mean_squared_error(y_true, y_pred):
    """ Returns the root mean squared error between y_true and y_pred """
    rmse = np.sqrt(np.mean(np.power(y_true - y_pred, 2)))
    return rmse


def max_error(y_true, y_pred):
    """ Returns maximum residual error """
    return np.max(np.abs(y_true - y_pred))


def sigmoid(x):
    """ Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))


def r2_score(y_true, y_pred):
    """
    r2 = 1 - (rss/ tss)
    rss = sum{i=0}^{n} (y_i - y_hat) ^ 2
    tss = sum{i=0}^{n} (y_i - y_bar) ^ 2
    """
    y_values = y_true.values
    y_average = np.average(y_values)

    residual_sum_of_squares = 0
    total_sum_of_squares = 0

    for i in range(len(y_values)):
        residual_sum_of_squares += (y_values[i] - y_pred[i])**2
        total_sum_of_squares += (y_values[i] - y_average)**2

    return 1 - (residual_sum_of_squares/total_sum_of_squares)

