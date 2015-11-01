__author__ = 'nelson'

import theano
import theano.tensor as T
import numpy as np

class Net():

    def __init__(self, solverArgs):
        self.solverArgs = solverArgs
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.o = T.ivector('o')  # observed death or not, 1 is death, 0 is right censored
        self.index = T.lscalar('index') # batch index
        self.layers = []
        return

    # Stack layers
    def push_layer(self, l):
        self.layers.append(l)
        return self.layers

    # Set up layers dimensions
    def compile(self):

        prev_out_data = self.x
        prev_out_shape = self.solverArgs['input_shape']
        for l in self.layers:

            l.compile(prev_out_data, prev_out_shape)
            prev_out_data = l.output_data
            prev_out_shape = l.output_shape
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
    def corruption_level(self):
        return T.scalar('corruption')

    @property
    def updates(self):
        return [(param, param + gparam * self.solverArgs['lr_rate'])for param, gparam in zip(self.params, self.gparams)]

    @property
    def cost(self):
        return self.layers[-1].cost(self.o)

    def cost_eval(self, observed_input, x_input):
        return self.layers[-1].cost_eval(self.o, self.x, observed_input, x_input)

    # Return pretraining function
    @property
    def pretraining_functions(self):

        pretrain_fns = []
        for l in self.layers:
            if l.name == 'DenoisingAutoEncoderLayer':

                # get the cost and the updates list
                cost, updates = l.get_cost_updates(self.solverArgs['corruption_level'], self.solverArgs['lr_rate'])

                # compile the theano function

                fn = theano.function(
                    on_unused_input='ignore',
                    inputs=[
                        self.index,
                        theano.Param(T.scalar('corruption'), default=0.2),
                        theano.Param(T.scalar('lr'), default=0.1)
                    ],
                    outputs=cost,
                    updates=updates,
                    givens={
                        self.x: self.solverArgs['train_x']
                    }
                )
                # append `fn` to the list of functions
                pretrain_fns.append(fn)
        return pretrain_fns

    # Return training function
    @property
    def foward_backward_function(self):

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
    def prediction_function(self):
        prediction_fn = theano.function(
            on_unused_input='ignore',
            inputs=[self.index],
            outputs=self.layers[-1].output_data,
            givens={
                self.x: self.solverArgs['train_x'],
                self.o: self.solverArgs['train_observed']
            },
            name='prediction'
        )
        return prediction_fn

    def checkGradients(self, epoch):
        prevgrads = [grad for grad in self.grad_function(epoch)]

        # Define epsilon
        e = 0.001**2
        aprox_grads = []
        for l in self.layers[1:]:
            for params in l.params:
                aprox_param = np.zeros(params.get_value().shape)
                print params.get_value().shape
                if len(params.get_value().shape) == 2:
                    for i in range(params.get_value().shape[0]):
                        for j in range(params.get_value().shape[1]):

                            # Sum epsilon
                            params_aux = params
                            params_plus = np.array(params.get_value())
                            params_minus = np.array(params.get_value())

                            # Compute cost plus
                            params_plus[i, j] += e
                            l.set_params(params_plus)
                            self.compile()
                            cost_plus = self.foward_backward_function(epoch)

                            # Compute cost minus
                            params_minus[i, j] -=  e

                            l.set_params(params_minus)
                            self.compile()
                            cost_minus = self.foward_backward_function(epoch)

                            # Put params back
                            l.set_params(params_aux)
                            self.compile()

                            # Compute gradient numerically
                            grad = (cost_plus - cost_minus) / 2 * e
                            aprox_param[i, j] = grad
                            print grad
                    aprox_grads.append(aprox_param)

                elif len(params.get_value().shape) == 1:
                    for i in range(params.get_value().shape[0]):
                        # Sum epsilon
                        params_aux = params
                        params_plus = np.array(params.get_value())
                        params_minus = np.array(params.get_value())

                        # Compute cost plus
                        params_plus[i] += e
                        l.set_params(params_plus)
                        self.compile()
                        cost_plus = self.foward_backward_function(epoch)

                        # Compute cost minus
                        params_minus[i] -=  e

                        l.set_params(params_minus)
                        self.compile()
                        cost_minus = self.foward_backward_function(epoch)

                        # Put params back
                        l.set_params(params_aux)
                        self.compile()

                        # Compute gradient numerically
                        grad = (cost_plus - cost_minus) / 2 * e
                        aprox_param[i] = grad
                        print grad
                        aprox_grads.append(aprox_param)

        # Compute difference
        diffs = [grad - appro_grad for grad, appro_grad in zip(self.grad_function(epoch), aprox_grads)]
        diffs2 = [grad - prevgrad for grad, prevgrad in zip(self.grad_function(epoch), prevgrads)]

        print "Diffs: " + str([d for d in diffs])
        print "Diffs2: " + str([d for d in diffs2])

        return

    @property
    def grad_function(self):
        grad_fn = theano.function(
            on_unused_input='ignore',
            inputs=[self.index],
            outputs=self.gparams,
            givens={
                self.x: self.solverArgs['train_x'],
                self.o: self.solverArgs['train_observed']
            },
            name='grad')
        return grad_fn