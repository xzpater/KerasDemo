import numpy as np


class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, Y):
        # X: N*D, every row is a image
        self.X = X
        self.Y = Y

    def predict(self, X):
        num_test = X.shape[0]
        y_predicted = np.zeros(num_test, dtype=self.Y.dtype)
        for i in range(num_test):
            distances = np.sum(np.abs(self.X - X[i, :]), axis=1)
            min_index = np.argmin(distances)
            y_predicted = y_predicted[min_index]
        return y_predicted
