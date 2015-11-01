__author__ = 'nelson'

import theano
from layers.plainLayer import PlainLayer
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.extra_ops as Te
import matlab

class CoxLossLayer(PlainLayer):

    def __init__(self, args):
        PlainLayer.__init__(self, 'CoxLossLayer')

        # Store arguments
        self.args = args
        return

    def compile(self, input_data, input_shape):

        # Store input data
        self.input_data = input_data

        # Compute shapes
        self.input_shape = input_shape
        self.output_shape = [input_shape[0], self.args['n_out']]

        # Generate a random state
        rng = np.random.RandomState(89677)

        # Initialize weights
        if self.W is None:
            self.W = theano.shared(
                value=np.asarray(
                    rng.uniform(
                        low=-np.sqrt(6. / (self.input_shape[1] + self.output_shape[1])),
                        high=np.sqrt(6. / (self.input_shape[1] + self.output_shape[1])),
                        size=(self.input_shape[1], self.output_shape[1])
                    ),
                    dtype=theano.config.floatX),name='W',borrow=True)

        # Store parameters
        self.params = [self.W]

        # Compute output
        self.output_data = T.dot(self.input_data, self.W).flatten()
        return


    def cost(self, observed):

        # Compute cost
        at_risk = self.args['at_risk']
        prediction = self.output_data
        exp = T.exp(prediction)[::-1]
        partial_sum = Te.cumsum(exp)[::-1]   # get the reversed partial cumulative sum
        log_at_risk = T.log(partial_sum[at_risk])
        diff = prediction - log_at_risk
        cost = T.sum(T.dot(observed, diff))
        return cost

    def coxInit(self, prev_input):

        # Cox initialization
        last_out_fn = theano.function(
            on_unused_input='ignore',
            inputs=[self.index],
            outputs=prev_input,
            givens={
                self.x: self.args['train_x'],
                self.o: self.args['train_observed']
            },
            name='last_output'
        )

        last_out = last_out_fn(0)
        eng = matlab.engine.start_matlab()
        cox_x = matlab.double(last_out.tolist())
        cox_y = matlab.double(self.args['train_y'].tolist())
        cox_c = matlab.double((1 - self.args['train_observed']).tolist())
        b = eng.coxphfit(cox_x, cox_y, 'censoring', cox_c)
        b = np.asarray([[w[0] for w in b]]).T
        self.reset_weight(b)
        print np.dot(last_out, b)

    def reset_weight(self, W_new):
        self.W.set_value(W_new)

    def set_params(self, params):
        self.params = params

if __name__ == '__main__':

    # Debbug
    coxArgs = {
        'train_x': np.array([[1,2],[3,4]]),
        'train_y': [1,2],
        'train_observed': [1, 0],
        'at_risk': [1, 0],
        'n_out': 1
    }
    coxLayer = CoxLossLayer(coxArgs)
    x = T.matrix('x')
    input_shape = [2, 2]
    coxLayer.compile(x, input_shape)
    print