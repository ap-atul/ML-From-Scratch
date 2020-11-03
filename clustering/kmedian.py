import numpy as np


def euclidean(points, center):
    distance = 0

    for i in range(len(points)):
        distance += (points[i] - center[i]) ** 2

    return np.sqrt(distance)


class KMedian:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

        self.classifications = dict()
        self.centroids = dict()

    def fit(self, data):
        # initializing the centroids to 1st 3 ele
        for i in range(self.k):
            self.centroids[i] = data[i]
            print(self.centroids)

        # iterate for max _iterations
        for _ in range(self.max_iter):

            for i in range(self.k):
                self.classifications[i] = []

            # create cluster & cal Euclidean distance
            for featureset in data:
                distances = [euclidean(featureset, self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            # recal median/centroid
            for classification in self.classifications:
                self.centroids[classification] = np.median(self.classifications[classification], axis=0)

            optimized = True

            # check for tolerance
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroi = self.centroids[c]
                if np.sum((current_centroi - original_centroid) / original_centroid * 100.0) > self.tol:
                    print("New centroid :: ", np.sum((current_centroi - original_centroid) / original_centroid * 100.0))
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
