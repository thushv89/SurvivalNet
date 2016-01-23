from loadMatData import *
from train import train
import numpy as np
import theano


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


def brain_wrapper(step=7):
    data_path = 'C:/Users/Song/Research/biomed/Survival/RSF/shuffle'

    val_scores = []
    for i in xrange(10):
        print '\n*** shuffling data iteration %d ***\n' % i
        p = data_path + str(i) + '.mat'
        O, X, T = load_mat_data(p=p, sort=False)
        X = np.asarray(X, dtype=theano.config.floatX)
        T = np.asarray(T, dtype=theano.config.floatX)

        fold_size = int(15 * len(X) / 100)

        # split data
        X_test = X[:fold_size]
        T_test = T[:fold_size]
        O_test = O[:fold_size]

        X_val = X[fold_size:2 * fold_size]
        T_val = T[fold_size:2 * fold_size]
        O_val = O[fold_size:2 * fold_size]

        X_train = X[fold_size * 2:]
        T_train = T[fold_size * 2:]
        O_train = O[fold_size * 2:]

        X_train, O_train = discrete_time_data(X_train, O_train, T_train, start=step, sort=True)

        X_val = np.hstack((X_val, np.asarray([T_val]).T))
        X_test = np.hstack((X_test, np.asarray([T_test]).T))

        print X_val.shape, X_test.shape

        validate_c = train(x_train=X_train, o_train=O_train, t_test=T_test, x_test=X_test, o_test=O_test, t_val=T_val,
                           o_val=O_val, x_val=X_val, testing=False, learning_rate=1e-6, n_hidden=15, L2_reg=0.075)
        val_scores.append(validate_c)

    print "\n*** Final Info ***\n"
    print 'average validate', np.mean(val_scores)
    print 'std validate', np.std(val_scores)


if __name__ == '__main__':
    brain_wrapper()
