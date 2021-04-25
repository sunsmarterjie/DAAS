# import matplotlib.pyplot as plt
import numpy as np
import math

EP = 50


def const():
    return 1


def linear(epoch):
    return max(0, 1 / (EP - 1) * (epoch - 1))


def log(epoch):
    return math.log(epoch, EP)


def exp(epoch):
    return 2 ** (epoch / 5) / (2 ** ((EP / 5) - 1)) / 2


def step(epoch):
    if epoch < EP / 4:
        return 0
    if epoch < EP / 2:
        return 1 / 3
    if epoch < EP * 3 / 4:
        return 2 / 3
    else:
        return 1


def sig(epoch):
    scale = 5
    return 1 / (1 + np.exp(-(epoch / scale - EP / (scale * 2))))


E = 8


def const_(epoch):
    return 0 if epoch <= E else 1


def linear_(epoch, EP=50 - E):
    return 0 if epoch <= E else 1 / (EP) * (epoch - E)


def log_(epoch):
    return 0 if epoch < E else math.log(epoch - E + 1, EP - E)


def exp_(epoch):
    return 0 if epoch <= E else 2 ** ((epoch - E) / 5) / (2 ** (((EP - E) / 5) - 1)) / 2


def sig_(epoch, EP=50):
    scale = 4
    return 0 if epoch <= E else 1 / (1 + np.exp(-((epoch - E) / scale - (EP - E) / (scale * 2))))


def step_(epoch):
    if epoch <= E:
        return 0
    if epoch < (EP - E) / 4 + E:
        return 0
    if epoch < (EP - E) / 2 + E:
        return 1 / 3
    if epoch < (EP - E) * 3 / 4 + E:
        return 2 / 3
    else:
        return 1


def show():
    x = np.linspace(1, EP, 1000)
    co = [const() for i in x]
    li = [linear(i) for i in x]
    lo = [log(i) for i in x]
    ex = [exp(i) for i in x]
    st = [step(i) for i in x]
    plt.plot(x, co, 'r')
    plt.plot(x, lo, 'b')
    plt.plot(x, st, 'pink')
    plt.plot(x, li, 'g')
    plt.plot(x, ex, 'c')
    plt.legend(['const', 'log', 'step', 'linear', 'exp'])

    plt.axis([1, EP + 1, 0, 1.2])
    plt.show()


if __name__ == '__main__':
    show()