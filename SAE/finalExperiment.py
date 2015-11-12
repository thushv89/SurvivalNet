"""
Created on Tue Oct 20 12:53:54 2015
@author: syouse3
"""
import os
from train import test_SdA
import math
import numpy as np
import matlab.engine
import scipy.io as sio
from loadMatData import VA


def shuffle_data(O, X, T):
    shuffled = np.insert(X, 0, O, axis=1)
    shuffled = np.insert(shuffled, 0, T, axis=1)
    np.random.shuffle(shuffled)
    O = shuffled[:, 1].astype(int)
    T = shuffled[:, 0].astype(int)
    X = shuffled[:, 2:]
    return O, X, T


def wrapper(shuffle_iter=10, k=10):
    p = VA
    mat = sio.loadmat(p)
    X = mat['X']
    X = X.astype('float64')
    O = mat['C']
    T = mat['T']

    T = np.asarray([t[0] for t in T])
    O = 1 - np.asarray([c[0] for c in O], dtype='int32')

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
        best_c_indices, first_c_indices, last_c_indices = cross_validation(C=O, X=X, T=T, m=m, F=F, k=k)
        best_c += best_c_indices
        first_c += first_c_indices
        last_c += last_c_indices

    print "\n*** Final Info ***\n"
    print "best: mean is %f, std is %f" % (float(np.mean(best_c)), float(np.std(best_c)))
    print "first: mean is %f, std is %f" % (float(np.mean(first_c)), float(np.std(first_c)))
    print "last: mean is %f, std is %f" % (float(np.mean(last_c)), float(np.std(last_c)))


def cross_validation(C, X, T, m, F, k):
    n_layer = 6
    hSize = 10
    do_rate = .1
    pretrain = True
    # store c indices
    best_c_indices = []
    first_c_indices = []
    last_c_indices = []

    cursor = -1
    foldn = 1

    while cursor < F * k:

        starti = int(cursor + 1)
        if m - cursor <= k:
            break
        else:
            endi = int(cursor + F)
        print starti
        print endi

        print "at cross validation index %d" % foldn

        x_train = np.concatenate((X[:starti], X[endi + 1:]), axis=0)
        t_train = np.concatenate((T[:starti], T[endi + 1:]), axis=0)
        c_train = np.concatenate((C[:starti], C[endi + 1:]), axis=0)

        x_test = X[starti:endi + 1]
        t_test = T[starti:endi + 1]
        c_test = C[starti:endi + 1]

        eng = matlab.engine.start_matlab()
        tmp = list(T)
        tmp = tmp[:starti] + tmp[endi + 1:]
        mat_T = matlab.double(tmp)
        sorted_t_train, order = eng.sort(mat_T, nargout=2)
        order = np.asarray(order[0]).astype(int) - 1
        sorted_t_train = matlab.double(sorted_t_train)
        at_risk = np.asarray(eng.ismember(sorted_t_train, sorted_t_train, nargout=2)[1][0]).astype(int)
        t_train = np.asarray(sorted_t_train[0])
        c_train = 1 - c_train[order]
        x_train = x_train[order]
        aa = at_risk - 1

        best, first, last = test_SdA(train_observed=c_train, train_X=x_train, train_y=t_train, at_risk_X=aa,
        test_observed = c_test, test_X=x_test, test_y=t_test, finetune_lr=0.01, pretrain=pretrain,
        pretraining_epochs=400, n_layers=n_layer,n_hidden = hSize, coxphfit=False, pretrain_lr=2.0,
        training_epochs=400, pretrain_mini_batch=True, batch_size=8, augment = False, drop_out=True,
        pretrain_dropout=False, dropout_rate=do_rate, grad_check=False, plot=False)

        best_c_indices.append(best)
        first_c_indices.append(first)
        last_c_indices.append(last)

        cursor += F
        foldn += 1

    return best_c_indices, first_c_indices, last_c_indices

if __name__ == '__main__':
    wrapper()
