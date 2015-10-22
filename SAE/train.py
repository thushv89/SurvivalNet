__author__ = 'Song'

import os
import sys
import timeit
import copy
from loadMatData import load_data, load_augment_data
from SdA import SdA
import numpy
from lifelines.utils import _naive_concordance_index
import matplotlib.pyplot as plt
import matlab.engine
import theano


def test_SdA(finetune_lr=0.01, pretraining_epochs=40, n_layers=3, n_hidden=140,
             pretrain_lr=1.0, training_epochs=200, batch_size=2, augment=False,
             drop_out=True, pretrain_dropout=False, dropout_rate=0.3):
    # observed, X, survival_time, at_risk_X = load_data('C:/Users/Song/Research/biomed/Survival/trainingData.csv')
    if augment:
        train_X, train_y, train_observed, at_risk_X, test_X, test_y, test_observed = load_augment_data()
    else:
        observed, X, survival_time, at_risk_X = load_data()
        test_size = len(X) / 3
        train_X = X[test_size:]
        train_y = survival_time[test_size:]
        train_observed = observed[test_size:]
        test_observed = observed[:test_size]
        test_X = X[:test_size]
        test_y = survival_time[:test_size]
    n_ins = train_X.shape[1]
    n_train_batches = len(train_X) / batch_size
    # changed to theano shared variable in order to do minibatch
    train_X = theano.shared(value=train_X, name='train_X')
    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=n_ins,
        hidden_layers_sizes=[n_hidden] * n_layers,
        n_outs=1,
        drop_out=drop_out,
        pretrain_dropout=pretrain_dropout,
        dropout_rate=dropout_rate,
        at_risk=at_risk_X
    )

    # sda_plus = SdA(
    #     numpy_rng=numpy_rng,
    #     n_ins=n_ins,
    #     hidden_layers_sizes=[140] * n_layers,
    #     n_outs=1,
    #     at_risk=at_risk_X
    # )
    #
    # sda_minus = SdA(
    #     numpy_rng=numpy_rng,
    #     n_ins=n_ins,
    #     hidden_layers_sizes=[140] * n_layers,
    #     n_outs=1,
    #     at_risk=at_risk_X
    # )

    # end-snippet-3 start-snippet-4
    #########################
    # PRETRAINING THE MODEL #
    #########################
    print '... getting the pretraining functions'
    pretraining_fns = sda.pretraining_functions(train_set_x=train_X,
                                                batch_size=batch_size)

    print '... pre-training the model'
    start_time = timeit.default_timer()
    # Pre-train layer-wise
    # de-noising level, set to zero for now
    corruption_levels = [.0] * n_layers
    for i in xrange(sda.n_layers):
        # go through pretraining epochs
        for epoch in xrange(pretraining_epochs):
            # go through the training set
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index,
                         corruption=corruption_levels[i],
                         lr=pretrain_lr))
            print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
            print numpy.mean(c)

    end_time = timeit.default_timer()

    print >> sys.stderr, ('The pretraining code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))
    # end-snippet-4
    ########################
    # FINETUNING THE MODEL #
    ########################

    # get the training, validation and testing function for the model
    print '... getting the finetuning functions'
    train_fn, output_fn, grad_fn, last_out_fn = sda.build_finetune_functions(
        train_X=train_X,
        train_observed=train_observed,
        test_X=test_X,
        learning_rate=finetune_lr
    )
    # train_fn_plus, output_fn_plus, grad_fn_plus = sda_plus.build_finetune_functions(
    #     train_X=train_X,
    #     train_observed=train_observed,
    #     test_X=test_X,
    #     learning_rate=finetune_lr
    # )
    # train_fn_minus, output_fn_minus, grad_fn_minus = sda_minus.build_finetune_functions(
    #     train_X=train_X,
    #     train_observed=train_observed,
    #     test_X=test_X,
    #     learning_rate=finetune_lr
    # )

    # cox initialization
    last_out = last_out_fn(0)
    eng = matlab.engine.start_matlab()
    cox_x = matlab.double(last_out.tolist())
    cox_y = matlab.double(train_y.tolist())
    cox_c = matlab.double((1 - train_observed).tolist())
    b = eng.coxphfit(cox_x, cox_y, 'censoring', cox_c)
    b = numpy.asarray([[w[0] for w in b]]).T
    sda.logLayer.reset_weight(b)
    # print numpy.dot(last_out, b)

    print '... finetunning the model'
    # early-stopping parameters
    c = []
    cost_list = []
    epoch = 0
    while epoch < training_epochs:
        epoch += 1
        avg_cost = train_fn(epoch)
        test_harzard = output_fn(epoch)
        # grad = grad_fn(epoch)
        # parameter = [param.get_value() for param in sda.params]
        # print parameter[-1]
        # gradient_check(grad, parameter, sda_plus, sda_minus, train_fn_plus, train_fn_minus)
        c_index = _naive_concordance_index(test_y, test_harzard, test_observed)
        c.append(c_index)
        cost_list.append(avg_cost)
        print 'at epoch %d, cost is %f, test c_index is %f' % (epoch, avg_cost, c_index)
    # plt.ylim(0.2, 0.8)
    print 'best score is: %f' % max(c)
    plt.plot(range(len(c)), c, c='r', marker='o', lw=5, ms=10, mfc='c')
    plt.show()
    plt.plot(range(len(cost_list)), cost_list, c='r', marker='o', lw=5, ms=10, mfc='c')
    plt.show()


def gradient_check(grad, parameter, sda_plus, sda_minus, train_fn_plus, train_fn_minus, e=0.01**2, epoch=1):
    for l in xrange(len(parameter)):
        param = parameter[l]
        if len(param.shape) == 1:
            for i in xrange(len(param)):
                # dimention level
                param_plus = copy.copy(param)
                param_plus[i] += e
                param_minus = copy.copy(param)
                param_minus[i] -= e
                # layer level
                parameter_plus = copy.copy(parameter)
                parameter_plus[l] = parameter_plus
                parameter_minus = copy.copy(parameter)
                parameter_minus[l] = parameter_minus
                # reset weight
                sda_plus.reset_weight(parameter_plus)
                sda_minus.reset_weight(parameter_minus)
                # get reset cost
                cost_plus = train_fn_plus(epoch)
                cost_minus = train_fn_minus(epoch)
                appro_grad = (cost_plus - cost_minus) / 2 * e
                diff = grad[l][i] - appro_grad
                print diff
        if len(param.shape) == 2:
            for i in xrange(len(param)):
                for j in xrange(len(param[i])):
                    # dimention level
                    param_plus = copy.copy(param)
                    param_plus[i][j] += e
                    param_minus = copy.copy(param)
                    param_minus[i][j] -= e
                    # layer level
                    parameter_plus = copy.copy(parameter)
                    parameter_plus[l] = param_plus
                    parameter_minus = copy.copy(parameter)
                    parameter_minus[l] = param_minus
                    # reset weight
                    sda_plus.reset_weight(parameter_plus)
                    sda_minus.reset_weight(parameter_minus)
                    # get reset cost
                    cost_plus = train_fn_plus(epoch)
                    cost_minus = train_fn_minus(epoch)
                    appro_grad = (cost_plus - cost_minus) / (2 * e)
                    diff = grad[l][i][j] - appro_grad
                    print diff / grad[l][i][j] * 100

if __name__ == '__main__':
    test_SdA()