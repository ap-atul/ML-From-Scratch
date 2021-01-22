from math import sqrt

def euclidean(a, b):
    return sqrt(sum([(a_i - b_i) ** 2 for a_i, b_i in zip(a, b)]))

def accuracy(act, pred):
    return (sum([1 for yt, yp in zip(act, pred) if yt == yp]) / len(pred))

class Knn:
    def __init__(self):
        self.xtrain, self.ytrain = None, None

    def fit(self, X_train, y_train):
        self.xtrain, self.ytrain = X_train, y_train

    def predict(self, X_test, k=5):
        return [self.closet(row, k) for row in X_test]

    def closet(self, row, k):
        distances = [(i, euclidean(row, self.xtrain[i])) for i in range(len(self.xtrain))]
        distances = sorted(distances, key=lambda x: x[1])[0: k]
        k_indices = [distances[i][0] for i in range(k)]
        k_labels = [self.ytrain[k_indices[i]] for i in range(k)]
        return max(set(k_labels), key=k_labels.count)
