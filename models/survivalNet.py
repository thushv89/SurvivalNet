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
                'n_out': 2,
                'W': solverArgs['W'],
                'b': solverArgs['b'],

            }
            self.push_layer(InnerProductLayer(innerProductArgs))

            # Add non-linear function
            nonLinearArgs = {
                'activation': ReLU,
                'alpha': None
            }
            self.push_layer(NonLinearLayer(nonLinearArgs))


        # Set loss funtion CoxLossLayer
        coxArgs = {
            'train_x': self.solverArgs['train_x'],
            'train_y': self.solverArgs['train_y'],
            'train_observed': self.solverArgs['train_observed'],
            'at_risk': solverArgs['at_risk_x'],
            'n_out': 1
        }
        coxLayer = CoxLossLayer(coxArgs)
        self.push_layer(coxLayer)

        # Compile network
        self.compile()

        # Initialize Cox
        #coxLayer.coxInit(self.solverArgs['input'])
        return

    def evaluate(self, test_y, out, test_x):
        return _naive_concordance_index(test_y, out, test_x)

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