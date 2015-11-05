__author__ = 'nelson'

from engine.net import Net
from layers.dataLayer import DataLayer
from layers.innerProductLayer import InnerProductLayer
from layers.denoisingLayer import DenoisingAutoEncoderLayer
from layers.coxLossLayer import CoxLossLayer
import numpy as np
import theano.tensor as T
from lifelines.utils import _naive_concordance_index
import timeit
from layers.nonLinearLayer import NonLinearLayer

class StackedAutoEncoders(Net):

    def __init__(self, solverArgs):
        Net.__init__(self, solverArgs)

        # Add DataLayer
        dataArgs = {
            'input_data': self.x,
            'input_shape': solverArgs['input_shape']
        }
        self.push_layer(DataLayer(dataArgs))

        # Add N InnerProductLayers and N DenoisingAutoEncoderLayer
        for _ in range(solverArgs['n_layers']):
            innerProductArgs = {
                'rng': np.random.RandomState(1234),
                'n_out': solverArgs['n_out'],
                'W': solverArgs['W'],
                'b': solverArgs['b']
            }
            self.push_layer(InnerProductLayer(innerProductArgs))

            # Add non-linear function
            nonLinearArgs = {
                'activation': T.nnet.sigmoid,
                'alpha': None
            }
            self.push_layer(NonLinearLayer(nonLinearArgs))

            daArgs = {
                'numpy_rng': solverArgs['numpy_rng'],
                'n_hidden' : solverArgs['n_out'],
                'W' : innerProductArgs['W'],
                'bhid' : innerProductArgs['b'],
                'bvis' : None,
                'n_out' : solverArgs['n_out']
            }
            self.push_layer(DenoisingAutoEncoderLayer(daArgs))

        # Compile network
        self.compile()
        return

    def evaluate(self, test_y, out, test_x):
        return _naive_concordance_index(test_y, out, test_x)

    def train(self):

        print '... pre-training the model'

        # Pre-train layer-wise
        # de-noising level, set to zero for now
        corruption_levels = [.0] * self.solverArgs['n_layers']
        pretraining_functions = self.pretraining_functions
        for i in xrange(self.solverArgs['n_layers']):
            # go through pretraining epochs
            for epoch in xrange(self.solverArgs['iters']):
                # go through the training set
                c = []
                for batch_index in range(self.solverArgs['n_train_batches']):
                    c.append(pretraining_functions[i](index=batch_index,
                             corruption=corruption_levels[i],
                             lr=self.solverArgs['lr_rate']))

                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print np.mean(c)

        return



