__author__ = 'nelson'

from engine.net import Net
from layers.dataLayer import DataLayer
from layers.innerProductLayer import InnerProductLayer
from layers.coxLossLayer import CoxLossLayer
import numpy as np
import theano.tensor as T
from lifelines.utils import _naive_concordance_index
from layers.nonLinearLayer import NonLinearLayer
import matplotlib as plt
from utils.nonLinearities import ReLU, leakyReLU
from layers.denoisingLayer import DenoisingLayer

class SurvivalNet(Net):

    def __init__(self, solverArgs):
        Net.__init__(self, solverArgs)

        # Add DataLayer
        dataArgs = {
            'input_data': self.x,
            'input_shape': self.solverArgs['input_shape']
        }
        self.push_layer(DataLayer(dataArgs))

        # Add N InnerProductLayers
        for _ in range(solverArgs['n_hidden_layers']):
            innerProductArgs = {
                'rng': np.random.RandomState(1234),
                'n_out': 200,
                'W': solverArgs['W'],
                'b': solverArgs['b'],

            }
            inLayer = InnerProductLayer(innerProductArgs)
            self.push_layer(inLayer)

            # Add non-linear function
            nonLinearArgs = {
                'activation': ReLU,
                'alpha': None
            }
            self.push_layer(NonLinearLayer(nonLinearArgs))

            # Set reconstruction layer
            daArgs = {
                'numpy_rng':np.random.RandomState(89677),
                'n_hidden' : 200,
                'W' : inLayer.W,
                'bhid' : inLayer.b,
                'bvis' : None,
                'n_out' : 200
            }
            self.push_layer(DenoisingLayer(daArgs))


        # Set loss funtion CoxLossLayer
        coxArgs = {
            'train_x': self.solverArgs['train_x'],
            'train_y': self.solverArgs['train_y'],
            'train_observed': self.solverArgs['train_observed'],
            'at_risk': solverArgs['at_risk_x'],
            'index': self.index,
            'x': self.x,
            'o': self.o,
            'n_out': 1
        }
        coxLayer = CoxLossLayer(coxArgs)
        self.push_layer(coxLayer)

        # Compile network
        self.compile()

        return

    def evaluate(self, test_y, out, test_x):
        return _naive_concordance_index(test_y, out, test_x)

    def pretrain(self):
        print '... pre-training the model'

        # Pre-train layer-wise
        # de-noising level, set to zero for now
        corruption_levels = [.0] * self.solverArgs['n_hidden_layers']
        pretraining_functions = self.pretraining_functions
        for i in xrange(self.solverArgs['n_hidden_layers']):
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

    def train(self):

        c = []
        cost_list = []
        res = []
        for epoch in range(self.solverArgs['iters']):

            avg_cost = self.foward_backward_update_function(epoch)
            test_harzard = self.prediction_function(epoch)
            res.append(test_harzard)

            # Check gradients numerically
            #self.checkGradients(epoch)

            # Evaluate model
            c_index = self.evaluate(self.solverArgs['test_y'], test_harzard, self.solverArgs['test_observed'])

            # Store avg_cost and c_index
            cost_list.append(avg_cost)
            c.append(c_index)

            # Print state
            print 'at epoch %d, cost is %f, test c_index is %f' % (epoch, avg_cost, c_index)

        #plt.plot(range(len(c)), c, c='r', marker='o', lw=5, ms=10, mfc='c')
        #plt.show()
        #plt.plot(range(len(cost_list)), cost_list, c='r', marker='o', lw=5, ms=10, mfc='c')
        #plt.show()
        return