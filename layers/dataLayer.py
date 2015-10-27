__author__ = 'nelson'

from layers.plainLayer import PlainLayer
import numpy as np

class DataLayer(PlainLayer):

    def __init__(self, args):
        self.args = args
        self.input_data = np.array(args['input'])
        self.output_data = np.array(args['input'])
        return

    def compile(self, _):
        return
