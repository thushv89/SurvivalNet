__author__ = 'nelson'

import numpy as np
import theano
import theano.tensor as T
from layers.plainLayer import PlainLayer

class InnerProductLayer(PlainLayer):

    def __init__(self, args):

        PlainLayer.__init__(self, 'InnerProductLayer')
        self.args = args

    def compile(self, input_data, input_shape, debbug=False):

        # Store input datas
        self.input_data = input_data

        # Compute shapes
        self.output_shape = [input_shape[0], self.args['n_out']] # [batch_size, out_size]
        self.input_shape = input_shape

        # Initialize weights
        if self.W is None:
            W_values = np.asarray(self.args['rng'].uniform(
                    low=-np.sqrt(6. / (self.input_shape[1] + self.output_shape[1])),
                    high=np.sqrt(6. / (self.input_shape[1] + self.output_shape[1])),
                    size=(self.input_shape[1], self.output_shape[1])), dtype=theano.config.floatX)
            if self.args['activation'] == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
            self.W = W

        # Initialize biases
        if self.b is None:
            b_values = np.zeros((self.args['n_out'],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            self.b = b

        # Store parameters
        self.params = [self.W, self.b]

        # Foward
        lin_output = T.dot(self.input_data, self.W) + self.b

        # Compute output
        self.output_data = (lin_output if self.args['activation'] is None else self.args['activation'](lin_output))

        # Debbug
        if debbug == True:
            print self.name
            print "input shape: " + str(self.input_shape)
            print "output shape: " + str(self.output_shape)
        return self.output_data

    def set_params(self, params):
        self.params = params