import numpy as np
from sklearn.metrics import confusion_matrix
import pickle


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

    def getConfusionMat(self, mu, y):
        mu = [0 if mui < 0.5 else 1 for mui in mu]
        c = confusion_matrix(y, mu)
        return c
        tp, fp, fn, tn = c[0, 0], c[1, 0], c[0, 1], c[1, 1]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        acc = (tp + tn) / (tp + fp + fn + tn)

    def update(self, X, y):
        mu = self.predict(X)
        R = np.zeros((mu.shape[0], mu.shape[0]))
        for i in range(mu.shape[0]):
            R[i, i] = mu[i] * (1 - mu[i])
        p1 = np.linalg.pinv(-np.dot(np.dot(np.transpose(X), R), X) - self.lam * np.eye(124))
        p2 = -self.lam * self.weight + np.dot(np.transpose(X), y - mu)
        self.weight = self.weight - np.dot(p1, p2)

        return np.sqrt(np.sum(np.square(self.weight)))

    def validation(self, X, y):
        mu = self.predict(X)
        c = self.getConfusionMat(mu, y)
        return c


if __name__ == '__main__':
    train_x, train_y = readFile('a9a/a9a')
    test_x, test_y = readFile('a9a/a9a.t')

    for l in [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        irls = IRLS(l)
        rec = {'l2norm': [], 'train_c': [], 'test_c': []}
        for i in range(30):
            print('lambda:', l, '  epoch:', i)
            l2norm = np.sqrt(np.sum(np.square(irls.weight)))
            train_c = irls.validation(train_x, train_y)
            test_c = irls.validation(test_x, test_y)
            rec['l2norm'].append(l2norm)
            rec['train_c'].append(train_c)
            rec['test_c'].append(test_c)
            irls.update(train_x, train_y)
        with open('figure_data/lam' + str(l) + '.pkl', 'wb') as f:
            pickle.dump(rec, f)
