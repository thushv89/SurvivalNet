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


def wrapper():
    pathout = '/home/syouse3/git/PySurv1.0/SAE/results/Brain-Oct28/FinalExperiment22l150hs/'
    
    if not os.path.exists(pathout):
        os.makedirs(pathout)

    p = VA
    mat = sio.loadmat(p)
    X = mat['X']
    C = mat['C']
    TT = mat['T']     
    print X.dtype
    X = X.astype('float64')
    print X.dtype
    print C.shape
    print TT.shape
    C = np.asarray([c[0] for c in C], dtype='int32')
    T = np.asarray([t[0] for t in TT], dtype='int32')
    print C.shape
    print T.shape
    
    layers = [22]
    hSizes = [150]
    do_rates = [.3]
    pt_ops = [True]

    k = 10
    m = len(X)
    # print 'm = %d' %(m)
    # F = math.floor(m/k)
    # print 'foldSize = %d' %(F)
    # cursor = -1
    # foldn = 1
    #
    # while cursor < F * k:
    #
    #     starti = int(cursor + 1)
    #     if m - cursor <= k:
    #         break
    #     else:
    #         endi = int(cursor + F)
    #     print starti
    #     print endi
    #
    #     x_train = np.concatenate((X[:starti], X[endi + 1:]), axis=0)
    #     t_train = np.concatenate((T[:starti], T[endi + 1:]), axis=0)
    #     c_train = np.concatenate((C[:starti], C[endi + 1:]), axis=0)
    #
    #     x_test = X[starti:endi + 1]
    #     t_test = T[starti:endi + 1]
    #     c_test = C[starti:endi + 1]
    #
    #     print 'train set'
    #     print t_train.shape
    #     print c_train.shape
    #     print x_train.shape
    #
    #     print 'test set:'
    #     print t_test.shape
    #     print c_test.shape
    #     print x_test.shape
    #
    #     eng = matlab.engine.start_matlab()
    #     tmp = [t[0] for t in TT]
    #     tmp = tmp[:starti] + tmp[endi + 1:]
    #     mat_T = matlab.double(tmp)
    #     sorted_t_train, order = eng.sort(mat_T, nargout=2)
    #     order = np.asarray(order[0]).astype(int) - 1
    #     sorted_t_train = matlab.double(sorted_t_train)
    #     at_risk = np.asarray(eng.ismember(sorted_t_train, sorted_t_train, nargout=2)[1][0]).astype(int)
    #     t_train = np.asarray(sorted_t_train[0])
    #     c_train = 1 - c_train[order]
    #     x_train = x_train[order]
    #     aa = at_risk - 1
    #     for hSize in hSizes:
    #         for n_layer in layers:
    #             for pretrain in pt_ops:
    #                 for do_rate in do_rates:
    #                     test_SdA(train_observed = c_train, train_X = x_train, train_y = t_train, at_risk_X = aa,\
    #                     test_observed = c_test, test_X = x_test, test_y = t_test, K = k, foldnum = foldn, \
    #                     finetune_lr=0.01, pretrain=pretrain, pretraining_epochs=400, n_layers=n_layer,\
    #                     n_hidden = hSize, coxphfit=False, pretrain_lr=2.0, training_epochs=400, pretrain_mini_batch=True, batch_size=8, \
    #                     augment = False, drop_out = True, pretrain_dropout=False, dropout_rate= do_rate,grad_check=False, \
    #                     plot=False, resultPath = pathout)
    #     cursor += F
    #     foldn += 1
                    
if __name__ == '__main__':
    wrapper()
