__author__ = 'nelson'

import theano
from layers.plainLayer import PlainLayer
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.extra_ops as Te

class CoxLossLayer(PlainLayer):

    params = None
    def __init__(self, args):

        self.args = args
        return

    def compile(self, input):

        self.input_data = input
        n_in = self.input_data.shape.eval()[1]
        rng = np.random.RandomState(89677)

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=np.asarray(
                rng.uniform(
                    low=-np.sqrt(6. / (n_in + self.args['n_out'])),
                    high=np.sqrt(6. / (n_in + self.args['n_out'])),
                    size=(n_in, self.args['n_out'])
                ),
                # rng.normal(size=(n_in, n_out)),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.params = [self.W]
        self.output_data = T.dot(self.input_data, self.W).flatten()

    def cost(self, observed):
        at_risk = self.args['at_risk']
        prediction = self.output_data
        # check = theano.printing.Print('prediction')(prediction)
        exp = T.exp(prediction)[::-1]
        partial_sum = Te.cumsum(exp)[::-1]   # get the reversed partial cumulative sum
        log_at_risk = T.log(partial_sum[at_risk])
        diff = prediction - log_at_risk
        cost = T.sum(T.dot(observed, diff))
        return cost

    def gradient(self):
        prediction = self.output_data
        risk = T.exp(prediction)
        product = self.input_data * (risk * T.ones((1, self.input_data.shape[0])))
        numerator = Te.cumsum(product[::-1])[::-1][self.args['at_risk']]
        denominator = Te.cumsum(risk[::-1])[::-1][self.args['at_risk']] * T.ones((1, self.input_data.shape[0]))
        # numerator = numerator.flatten()
        # denominator = denominator.flatten()
        # gradient = T.dot(observed, self.input - (numerator / denominator))
        return None

    def reset_weight(self, W_new):
        self.W.set_value(W_new)
