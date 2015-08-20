#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Song'
import numpy

import theano
import theano.tensor as T

sharedX = (lambda X, name:
           theano.shared(numpy.asarray(X, dtype=theano.config.floatX), name=name))

class OutputLayer(object):
    def __init__(self, input, ht, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """
        # start-snippet-1
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=numpy.zeros(n_out, dtype=theano.config.floatX),
            name='b',
            borrow=True
        )
        self.Wt1 = sharedX(1, 'Wt1')
        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        ht_wt1 = T.pow(ht, self.Wt1).flatten()
        # ht_wt1 = ht * self.Wt1
        dot = T.exp(T.dot(input, self.W) + self.b).flatten()
        self.p_y_given_x = 1 / (1 + dot * ht_wt1)
        # parameters of the model
        self.params = [self.W, self.b, self.Wt1]

    def negative_log_likelihood(self):
        # cost_printed = theano.printing.Print('Cost : ')(self.p_y_given_x)
        return T.mean(T.log(self.p_y_given_x))
        # end-snippet-2
