__author__ = 'nelson'

import numpy as np
import theano
import theano.tensor as T
from layers.plainLayer import PlainLayer

class NonLinearLayer(PlainLayer):

    def __init__(self, args):

        PlainLayer.__init__(self, 'NonLinearLayer')
        self.args = args
        self.a = args['alpha'] # Alpha parameter
        self.opt = False
        return

    def compile(self, input_data, input_shape):

        # Define shape
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.input_data = input_data

        # Compute output
        self.act = self.args['activation']
        if self.act == T.tanh:
            self.output_data = self.act(self.input_data)
        else:
            self.output_data = self.act(self.input_data, self.a)

        return self.output_data