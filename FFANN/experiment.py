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


def cross_validation(O, X, T, m, F, k=10, step=7):
    cursor = -1
    fold_n = 1
    # store c indices
    best_c_indices = []
    first_c_indices = []
    last_c_indices = []

    while cursor < F * k:
        start_i = int(cursor + 1)
        if m - cursor <= k:
            break
        else:
            end_i = int(cursor + F)

        print "at cross validation index %d" % fold_n

        x_train = np.concatenate((X[:start_i], X[end_i + 1:]), axis=0)
        t_train = np.concatenate((T[:start_i], T[end_i + 1:]), axis=0)
        o_train = np.concatenate((O[:start_i], O[end_i + 1:]), axis=0)

        x_train, o_train = discrete_time_data(x_train, o_train, t_train, start=step, sort=True)
        x_test = X[start_i:end_i + 1]
        t_test = T[start_i:end_i + 1]
        o_test = O[start_i:end_i + 1]

        best, first, last = train(x_train=x_train, o_train=o_train, t_test=t_test, x_test=x_test, o_test=o_test, print_info=False)
        best_c_indices.append(best)
        first_c_indices.append(first)
        last_c_indices.append(last)

        cursor += F
        fold_n += 1

    return best_c_indices, first_c_indices, last_c_indices


def wrapper(p, shuffle_iter=10, k=10):
    O, X, T = load_mat_data(p=p, sort=False)

    m = len(X)
    F = np.floor(m / k)
    print '*** data set size = %d, fold size = %d ***' % (m, F)
    print '*** fold size = %d *** \n' % F

    best_c = []
    first_c = []
    last_c = []
    for i in xrange(shuffle_iter):
        print '\n*** shuffling data iteration %d ***\n' % i
        O, X, T = shuffle_data(O, X, T)
        best_c_indices, first_c_indices, last_c_indices = cross_validation(O=O, X=X, T=T, m=m, F=F, k=k)
        best_c += best_c_indices
        first_c += first_c_indices
        last_c += last_c_indices

    print "\n*** Final Info ***\n"
    print "best: mean is %f, std is %f" % (float(np.mean(best_c)), float(np.std(best_c)))
    print "first: mean is %f, std is %f" % (float(np.mean(first_c)), float(np.std(first_c)))
    print "last: mean is %f, std is %f" % (float(np.mean(last_c)), float(np.std(last_c)))


if __name__ == '__main__':
    wrapper(p=VA)
