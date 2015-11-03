from loadMatData import *
from train import train
import numpy as np


def shuffle_data(O, X, T):
    shuffled = np.insert(X, 0, O, axis=1)
    shuffled = np.insert(shuffled, 0, T, axis=1)
    np.random.shuffle(shuffled)
    O = shuffled[:, 1].astype(int)
    T = shuffled[:, 0].astype(int)
    X = shuffled[:, 2:]
    return O, X, T


def cross_validation(O, X, T, k=10, step=7):
    m = len(X)
    print 'data set size = %d' % m
    F = np.floor(m / k)
    print 'fold size = %d' % F
    cursor = -1
    fold_n = 1

    # c indices
    c_indices = []

    while cursor < F * k:
        # print cursor
        # print F*k
        start_i = int(cursor + 1)
        if m - cursor <= k:
            break
        else:
            end_i = int(cursor + F)
        # print 'validation set index (%d, %d)' % (start_i, end_i)

        x_train = np.concatenate((X[:start_i], X[end_i + 1:]), axis=0)
        t_train = np.concatenate((T[:start_i], T[end_i + 1:]), axis=0)
        o_train = np.concatenate((O[:start_i], O[end_i + 1:]), axis=0)

        x_train, o_train = discrete_time_data(x_train, o_train, t_train, start=step, sort=True)
        x_test = X[start_i:end_i + 1]
        t_test = T[start_i:end_i + 1]
        o_test = O[start_i:end_i + 1]

        c_index = train(x_train=x_train, o_train=o_train, t_test=t_test, x_test=x_test, o_test=o_test)
        c_indices.append(c_index)
        cursor += F
        fold_n += 1
    return c_indices


def wrapper(p, shuffle_iter=10):
    O, X, T = load_mat_data(p=p, sort=False)
    c_indices = []
    for i in xrange(shuffle_iter):
        print 'shuffling data index at %d' % i
        O, X, T = shuffle_data(O, X, T)
        c_indices += cross_validation(O=O, X=X, T=T)
    print "mean is %f, std is %f" % (float(np.mean(c_indices)), float(np.std(c_indices)))

if __name__ == '__main__':
    wrapper(p=LUAD_P)
