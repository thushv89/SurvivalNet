__author__ = 'nelson'

import theano
import theano.tensor as T

class Net():
    x = T.matrix('x')  # the data is presented as rasterized images
    o = T.ivector('o')  # observed death or not, 1 is death, 0 is right censored
    layers = []
    solverArgs = None

    def __init__(self, solverArgs):
        self.solverArgs = solverArgs
        return

    # Stack layers
    def push_layer(self, l):
        self.layers.append(l)
        return self.layers

    # Set up layers dimensions
    def compile(self):
        prev_out = []
        for l in self.layers:
            l.compile(prev_out)
            prev_out = l.output_data
        return

    # Return all params from the net
    @property
    def params(self):
        params = []
        for l in self.layers[1:]:
            params.extend(l.params)
        return params

    # Return all gparams from the net
    @property
    def gparams(self):
        return T.grad(self.cost, self.params)

    @property
    def index(self):
        return T.lscalar('index')  # index to a [mini]batch

    @property
    def updates(self):
        return [(param, param + gparam * self.solverArgs['lr_rate'])for param, gparam in zip(self.params, self.gparams)]

    @property
    def cost(self):
        return self.layers[-1].cost(self.o)

    # Return training function
    @property
    def train_function(self):
        func = theano.function(
            on_unused_input='ignore',
            inputs=[self.index],
            outputs=self.cost,
            updates=self.updates,
            givens={
                self.x: self.solverArgs['train_x'],
                self.o: self.solverArgs['train_observed']
            },
            name='train')
        return func

    # Return testing function
    @property
    def output_function(self):
        output_fn = theano.function(
            on_unused_input='ignore',
            inputs=[self.index],
            outputs=self.layers[-1].output_data,
            givens={
                self.x: self.solverArgs['train_x'],
                self.o: self.solverArgs['train_observed']
            },
            name='output'
        )
        return output_fn

    # Reset weights
    def reset_weigths(self):
        return