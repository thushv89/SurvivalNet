__author__ = 'nelson'

from layers.plainLayer import PlainLayer
import numpy as np
import theano.tensor as T
class DataLayer(PlainLayer):

    def __init__(self, args):
        PlainLayer.__init__(self, 'DataLayer')
        self.args = args
        self.input_data = self.args['input_data']
        self.input_shape = np.array(args['input_shape'])

        return

    def compile(self, output_data, output_shape):

        self.output_data = output_data
        self.output_shape = np.array(output_shape)
        return
