__author__ = 'nelson'

__author__ = 'nelson'

from engine.net import Net
from layers import dataLayer, innerProductLayer, coxLossLayer, denoisingAutoencoderLayer
import numpy as np
import theano.tensor as T
from lifelines.utils import _naive_concordance_index

class preSurvivalNet(Net):

    net = None
    params = None

    def __init__(self, params):

        # Create Net
        net = Net(params.input, params.output)

        # Add InnerProductLayer
        innerProductParams = {
            'rng': np.random.RandomState(1234),
            'input': params.input,
            'n_in': 28 * 28,
            'n_out': 10,
            'W': params.W,
            'b': params.b,
            'activation': T.tanh
        }

        net.push_layer(denoisingAutoencoderLayer(innerProductParams))

        # Set loss funtion CoxLossLayer
        coxLossParam = {
        }
        net.push_layer(coxLossLayer(coxLossParam))

        return

    def pretrain(self):

        c = []
        cost_list = []
        for epoch in range(self.params.iters):

            avg_cost = self.net.train_function(epoch)
            loss_harzard = self.net.prediction_function(epoch)

            c_index = _naive_concordance_index(test_y, test_harzard, test_observed)
            c.append(c_index)
            cost_list.append(avg_cost)
            print 'at epoch %d, cost is %f, test c_index is %f' % (epoch, avg_cost, c_index)
        return




