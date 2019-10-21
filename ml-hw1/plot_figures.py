import pickle
import matplotlib.pyplot as plt
import numpy as np


def readData(filename):
    with open(filename, 'rb') as f:
        rec = pickle.load(f)
    return rec


def plot_l2norm(records):
    plt.figure()
    x = [i for i in range(1, 31)]
    plt.xlabel('iterations')
    plt.ylabel('L2 norm')
    for rec, l in zip(records, [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        plt.plot(x, rec['l2norm'], label='lambda=' + str(l))
        plt.legend()
    plt.savefig('figures/l2norm.png')
    plt.show()


def plot_accuracy(records):
    plt.figure()
    x = [i for i in range(1, 31)]
    plt.xlabel('iterations')
    plt.ylabel('Accuracy')
    for rec, l in zip(records, [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        if l not in [0, 0.01, 10]:
            continue
        train_c = rec['train_c']
        test_c = rec['test_c']
        train_acc = []
        for c in train_c:
            train_acc.append((c[0, 0] + c[1, 1]) / np.sum(np.sum(c)))
        plt.plot(x, train_acc, label='train, lambda='+str(l))
        test_acc = []
        for c in test_c:
            test_acc.append((c[0, 0] + c[1, 1]) / np.sum(np.sum(c)))
        plt.plot(x, test_acc, '--', label='test, lambda=' + str(l))
    plt.legend()
    plt.savefig('figures/train_test_acc.png')
    plt.show()


def plot_recall(records):
    plt.figure()
    x = [i for i in range(1, 31)]
    plt.xlabel('iterations')
    plt.ylabel('Recall')
    for rec, l in zip(records, [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        if l not in [0, 0.01, 10]:
            continue
        train_c = rec['train_c']
        test_c = rec['test_c']
        train_recall = []
        for c in train_c:
            train_recall.append(c[0, 0] / (c[0, 0] + c[0, 1]))
        plt.plot(x, train_recall, label='train, lambda=' + str(l))
        test_recall = []
        for c in test_c:
            test_recall.append(c[0, 0] / (c[0, 0] + c[0, 1]))
        plt.plot(x, test_recall, '--', label='test, lambda=' + str(l))
    plt.legend()
    plt.savefig('figures/train_test_recall.png')
    plt.show()


def plot_precision(records):
    plt.figure()
    x = [i for i in range(1, 31)]
    plt.xlabel('iterations')
    plt.ylabel('Precision')
    for rec, l in zip(records, [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        if l not in [0, 0.01, 10]:
            continue
        train_c = rec['train_c']
        test_c = rec['test_c']
        train_recall = []
        for c in train_c:
            train_recall.append(c[0, 0] / (c[0, 0] + c[1, 0]))
        plt.plot(x, train_recall, label='train, lambda=' + str(l))
        test_recall = []
        for c in test_c:
            test_recall.append(c[0, 0] / (c[0, 0] + c[1, 0]))
        plt.plot(x, test_recall, '--', label='test, lambda=' + str(l))
    plt.legend()
    plt.savefig('figures/train_test_precision.png')
    plt.show()


def plot_f1score(records):
    plt.figure()
    x = [i for i in range(1, 31)]
    plt.xlabel('iterations')
    plt.ylabel('F1-score')
    for rec, l in zip(records, [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]):
        if l not in [0, 0.01, 10]:
            continue
        train_c = rec['train_c']
        test_c = rec['test_c']
        train_f1score = []
        for c in train_c:
            p = c[0, 0] / (c[0, 0] + c[1, 0])
            r = c[0, 0] / (c[0, 0] + c[0, 1])
            train_f1score.append(2 * p * r / (p + r))
        plt.plot(x, train_f1score, label='train, lambda=' + str(l))
        test_f1score = []
        for c in test_c:
            p = c[0, 0] / (c[0, 0] + c[1, 0])
            r = c[0, 0] / (c[0, 0] + c[0, 1])
            test_f1score.append(2 * p * r / (p + r))
        plt.plot(x, test_f1score, '--', label='test, lambda=' + str(l))
    plt.legend()
    plt.savefig('figures/train_test_f1score.png')
    plt.show()


if __name__ == '__main__':
    records = []
    for l in [0, 0.001, 0.01, 0.1, 1, 10, 100, 1000]:
        rec = readData('figure_data/lam' + str(l) + '.pkl')
        records.append(rec)
    # plot_l2norm(records)
    # plot_accuracy(records)
    # plot_recall(records)
    # plot_precision(records)
    plot_f1score(records)