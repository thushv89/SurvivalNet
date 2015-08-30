#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = 'Song'
import numpy

import theano
import theano.tensor as T

class OutputLayer(object):
    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: all discrete time model data

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
        self.input = input
        self.hazard_ratio = T.nnet.sigmoid(T.dot(input, self.W) + self.b).flatten()
        self.params = [self.W, self.b]

    def cost(self, o):
        return T.sum(T.nnet.binary_crossentropy(self.hazard_ratio, o))
