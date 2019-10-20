import numpy as np


def readFile(filename):
    with open(filename) as f:
        labels = []
        features = []
        for line in f:
            line = line.split()
            labels.append(max(0, int(line[0])))
            line = line[1:]
            feature = np.zeros(124)
            feature[0] = 1.0
            for id in line:
                id = id.split(':')
                feature[int(id[0])] = int(id[1])
            features.append(feature)
    features = np.array(features)
    labels = np.array(labels)
    return features, labels


class IRLS:
    def __init__(self, lam=0.0):
        self.lam = lam
        self.weight = np.random.randn(124) * 0.01

    def predict(self, X):
        return 1 / (1 + np.e ** (-np.dot(X, self.weight)))

    def update(self, X, y):
        mu = self.predict(X)
        R = np.zeros((mu.shape[0], mu.shape[0]))
        for i in range(mu.shape[0]):
            R[i, i] = mu[i] * (1 - mu[i])
        p1 = np.linalg.pinv(-np.dot(np.dot(np.transpose(X), R), X) - self.lam * np.eye(124))
        p2 = self.lam * self.weight + np.dot(np.transpose(X), y-mu)
        self.weight = self.weight - np.dot(p1, p2)
        lw = 0
        for i in range(X.shape[0]):
            s = np.dot(self.weight, X[i, :])
            lw += y[i] * s - np.log(1 + np.exp(s))
        loss = -0.5 * np.sum(np.square(self.weight)) + lw
        print(loss)



if __name__ == '__main__':
    train_x, train_y = readFile('a9a/a9a')
    test_x, test_y = readFile('a9a/a9a.t')

    irls = IRLS()
    for i in range(100):
        irls.update(train_x, train_y)