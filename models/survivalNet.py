__author__ = 'nelson'

from engine.net import Net
from layers.dataLayer import DataLayer
from layers.innerProductLayer import InnerProductLayer
from layers.coxLossLayer import CoxLossLayer
import numpy as np
import theano.tensor as T
from lifelines.utils import _naive_concordance_index

class SurvivalNet(Net):

    net = None
    args = None

    def __init__(self, args):

        self.args = args

        # Create Net
        self.net = Net(args)

        # Add DataLayer
        dataArgs = {
            'input': args['train_x'],
            'batch_size': args['batch_size']
        }
        self.net.push_layer(DataLayer(dataArgs))

        # Add N InnerProductLayers
        for _ in range(args['n_hidden_layers']):
            innerProductArgs = {
                'rng': np.random.RandomState(1234),
                'n_out': 200,
                'W': args['W'],
                'b': args['b'],
                'activation': T.tanh
            }
            self.net.push_layer(InnerProductLayer(innerProductArgs))


        # Set loss funtion CoxLossLayer
        coxArgs = {
            'at_risk': args['at_risk_x'],
            'n_out': 1
        }
        self.net.push_layer(CoxLossLayer(coxArgs))

        # Compile network
        self.net.compile()
        return

    def evaluate(self, test_y, out, test_x):
        return _naive_concordance_index(test_y, out, test_x)

    def train(self):

        c = []
        cost_list = []
        for epoch in range(self.args['iters']):

            avg_cost = self.net.train_function(epoch)
            test_harzard = self.net.output_function(epoch)

            # Evaluate model
            c_index = self.evaluate(self.args['test_y'], test_harzard, self.args['test_observed'])

            # Store avg_cost and c_index
            cost_list.append(avg_cost)
            c.append(c_index)

            # Print state
            print 'at epoch %d, cost is %f, test c_index is %f' % (epoch, avg_cost, c_index)
        return




