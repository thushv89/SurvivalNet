__author__ = 'nelson'

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from layers.plainLayer import PlainLayer

class DenoisingLayer(PlainLayer):
    """Denoising Auto-Encoder class (dA)
    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.
    .. math::
        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)
        y = s(W \tilde{x} + b)                                           (2)
        x = s(W' y  + b')                                                (3)
        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)
    """

    def __init__( self, args):
        PlainLayer.__init__(self, 'DenoisingAutoEncoderLayer')
        self.args = args
        self.theano_rng = None
        self.opt = False
        return

    def compile(self, input_data, input_shape, debbug=False):

        # Store data
        self.input_data = input_data
        self.output_data = input_data

        # Compute shape
        self.input_shape = input_shape
        self.output_shape = input_shape

        # Create a Theano random generator that gives symbolic random values
        if not self.theano_rng:
            self.theano_rng = RandomStreams(self.args['numpy_rng'].randint(2 ** 30))

        # Initialize weights
        if not self.args['W']:

            initial_W = np.asarray(
                self.args['numpy_rng'].uniform(
                    low=-4 * np.sqrt(6. / (self.output_shape[1] + self.input_shape[1])),
                    high=4 * np.sqrt(6. / (self.output_shape[1] + self.input_shape[1])),
                    size=(self.output_shape[1], self.output_shape[1])),dtype=theano.config.floatX)
            W = theano.shared(value=initial_W, name='W', borrow=True)

        # Initialize bias
        if not self.args['bvis']:
            bvis = theano.shared(
                value=np.zeros(
                    self.input_shape[1],
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not self.args['bhid']:
            bhid = theano.shared(
                value=np.zeros(
                    self.output_shape[1],
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        # Store parameters
        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.params = [self.W, self.b, self.b_prime]

        # If no input is given, generate a variable representing the input
        if input_data is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.input_data = T.dmatrix(name='input')

        # Debbug
        if debbug == True:
            print self.name
            print "input shape: " + str(self.input_shape)
            print "output shape: " + str(self.output_shape)
        return

    def get_corrupted_input(self, corruption_level):
        """This function keeps ``1-corruption_level`` entries of the inputs the
        same and zero-out randomly selected subset of size ``coruption_level``
        Note : first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
               second argument is the number of trials
               third argument is the probability of success of any trial
                this will produce an array of 0s and 1s where 1 has a
                probability of 1 - ``corruption_level`` and 0 with
                ``corruption_level``
                The binomial function return int64 data type by
                default.  int64 multiplicated by the input
                type(floatX) always return float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, this don't change the
                result. This is needed to allow the gpu to work
                correctly as it only support float32 for now.
        """
        return self.theano_rng.binomial(size=self.input_shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * self.input_data

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the
        hidden layer
        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        tilde_x = self.get_corrupted_input(corruption_level)

        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        # L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        L = T.sum((self.input_data - z) ** 2, axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [(param, param - learning_rate * gparam)for param, gparam in zip(self.params, gparams) ]
        return (cost, updates)

