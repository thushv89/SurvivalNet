__author__ = 'Song'

import os
import sys
import timeit
from loadData import load_data
from SdA import SdA
import numpy
from lifelines.utils import _naive_concordance_index
import matplotlib.pyplot as plt


def test_SdA(finetune_lr=10.0, pretraining_epochs=30, n_layers=3,
             pretrain_lr=1.0, training_epochs=600, batch_size=1):

    observed, X, survival_time, at_risk_X = load_data('C:/Users/Song/Research/biomed/Survival/trainingData.csv')
    n_ins = X.shape[1]
    test_size = len(X) / 3
    train_X = X[test_size:]
    train_observed = observed[test_size:]
    test_observed = observed[:test_size]
    test_X = X[:test_size]
    test_y = survival_time[:test_size]
    n_train_batches = len(train_X) / batch_size
    # numpy random generator
    # start-snippet-3
    numpy_rng = numpy.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = SdA(
        numpy_rng=numpy_rng,
        n_ins=n_ins,
        hidden_layers_sizes=[180] * n_layers,
        n_outs=1,
        at_risk=at_risk_X
    )
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
    train_fn, output_fn = sda.build_finetune_functions(
        train_X=train_X,
        train_observed=train_observed,
        test_X=test_X,
        learning_rate=finetune_lr
    )

    print '... finetunning the model'
    # early-stopping parameters
    c = []
    epoch = 0
    while epoch < training_epochs:
        epoch += 1
        avg_cost = train_fn(epoch)
        test_harzard = output_fn(epoch)
        c_index = _naive_concordance_index(test_y, test_harzard, test_observed)
        c.append(c_index)
        print 'at epoch %d, cost is %f, test c_index is %f' % (epoch, avg_cost, c_index)
    plt.ylim(0.2, 0.8)
    plt.plot(range(len(c)), c, c='r', marker='o', lw=5, ms=10, mfc='c')
    plt.show()
if __name__ == '__main__':
    test_SdA()