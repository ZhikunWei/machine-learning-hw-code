import matplotlib.pyplot as plt

if __name__ == '__main__':
    x, y = [], []
    with open('log.txt') as f:
        for line in f:
            line = line.split()
            x.append(int(line[0]))
            y.append(float(line[1]))
    plt.plot(x, y)
    plt.xlabel('episode')
    plt.ylabel('score')
    plt.savefig('result.png')
    plt.show()