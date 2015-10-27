__author__ = 'nelson'

import numpy as np
import theano
import theano.tensor as T

class InnerProductLayer():

    W = None
    b = None
    args = None
    input_data = None
    output_data = None
    params = None
    gparams = None
    def __init__(self, args):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).
        NOTE : The nonlinearity used here is tanh
        Hidden unit activation is given by: tanh(dot(input,W) + b)
        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights
        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)
        :type n_in: int
        :param n_in: dimensionality of input
        :type n_out: int
        :param n_out: number of hidden units
        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.args = args

    def compile(self, input_data):
        self.input_data = input_data

        if isinstance(input_data, np.ndarray):
            n_in = self.input_data.shape[1]
        else:
            n_in = self.input_data.shape.eval()[1]

        if self.args['W'] is None:
            W_values = np.asarray(self.args['rng'].uniform(
                    low=-np.sqrt(6. / (n_in + self.args['n_out'])),
                    high=np.sqrt(6. / (n_in + self.args['n_out'])),
                    size=(n_in, self.args['n_out'])), dtype=theano.config.floatX)
            if self.args['activation'] == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if self.args['b'] is None:
            b_values = np.zeros((self.args['n_out'],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)


        self.W = W
        self.b = b
        self.params = [W, b]

        # Foward
        self.input_data = input_data
        lin_output = T.dot(input_data, self.W) + self.b

        self.output_data = (lin_output if self.args['activation'] is None else self.args['activation'](lin_output))
        return self.output_data