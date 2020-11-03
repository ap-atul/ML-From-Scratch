import numpy as np

TOL = 0.001
ITER = 300
CLUSTER = 2


# random no generator for 2D array
def _random(bound, size):
    _rv = []
    _vis = []
    while True:
        r = np.random.randint(bound)
        if r in _vis:
            pass
        else:
            _vis.append(r)
            _rv.append(r)

        if len(_rv) == size:
            return _rv


class KMedoids:
    def __init__(self, k=CLUSTER, tol=TOL, max_iter=ITER):
        self.classifications = {}
        self.medoids = {}
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        _med = _random(len(data), self.k)

        for i in range(self.k):
            # taking random
            self.medoids[i] = data[_med[i]]

            # taking average
            # self.medoids[i] = np.average(data[i], axis=0)
            print(self.medoids[i])

        # iterate for max _iterations
        for _ in range(self.max_iter):

            for i in range(self.k):
                self.classifications[i] = []

            # create cluster & cal Euclidean distance
            for featureset in data:
                distances = [np.linalg.norm(featureset - self.medoids[medoid]) for medoid in self.medoids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

    def predict(self, data):
        distances = [np.linalg.norm(data - self.medoids[medoid]) for medoid in self.medoids]
        classification = distances.index(min(distances))
        return classification
