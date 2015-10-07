__author__ = 'Song'
import numpy
import theano
import theano.tensor as T
import theano.tensor.extra_ops as Te


def test():
    m = T.ivector('m')
    v = T.ivector('v')
    sum = T.sum(m - v)
    f = theano.function([m, v], sum)
    print f([[1, 2, 3, 4], [1, 2, 3, 4]])


class LogLikelihoodLayer(object):
    def __init__(self, input, n_in, n_out):
        rng = numpy.random.RandomState(89677)
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                # rng.normal(size=(n_in, n_out)),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        # self.b = theano.shared(
        #     value=numpy.zeros(n_out, dtype=theano.config.floatX),
        #     name='b',
        #     borrow=True
        # )
        self.input = input
        self.output = T.dot(self.input, self.W).flatten()
        self.params = [self.W]

    def cost(self, observed, at_risk):
        prediction = self.output
        partial_sum = Te.cumsum(T.exp(prediction)[::-1])[::-1]    # get the reversed partial cumulative sum
        # partial_sum = T.set_subtensor(partial_sum[0], 0)    # set the last one to be zero
        cost = T.sum(T.dot(observed, prediction - T.log(partial_sum[at_risk])))
        return cost


if __name__ == '__main__':
    test()
    # logLayer = LogLikelihoodLayer(input=T.matrix('x'), n_in=2, n_out=1)
