import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter

kmf = KaplanMeierFitter()

import numpy
from scipy.special import expit
import theano
import theano.tensor as T

from output import OutputLayer, sharedX
from loadData import load_training_data


class HiddenLayer(object):
    def __init__(self, rng, input, t, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        self.input = input
        if W is None:
            W_values = numpy.asarray(
                rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b
        self.Wt2 = sharedX(1, 'Wt2')
        lin_output = T.dot(input, self.W) + self.b
        ht_output = (t * self.Wt2)
        self.output = activation(lin_output)
        self.ht_output = ht_output
        # parameters of the model
        self.params = [self.W, self.b, self.Wt2]


# start-snippet-2
class MLP(object):
    def __init__(self, rng, input, t, n_in, n_hidden, n_out):

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayer = HiddenLayer(
            rng=rng,
            input=input,
            t=t,
            n_in=n_in,
            n_out=n_hidden,
            activation=theano.tensor.nnet.sigmoid
        )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.OutputLayer = OutputLayer(
            input=self.hiddenLayer.output,
            ht=self.hiddenLayer.ht_output,
            n_in=n_hidden,
            n_out=n_out
        )
        # end-snippet-2 start-snippet-3
        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = (
            abs(self.hiddenLayer.W).sum() + abs(self.hiddenLayer.Wt2).sum()
            + abs(self.OutputLayer.W).sum() + abs(self.OutputLayer.Wt1).sum()
        )

        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small
        self.L2_sqr = (
            (self.hiddenLayer.W ** 2).sum() + (self.hiddenLayer.Wt2 ** 2).sum()
            + (self.OutputLayer.W ** 2).sum() + (self.OutputLayer.Wt1 ** 2).sum()
        )
        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.OutputLayer.negative_log_likelihood
        )
        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.OutputLayer.params
        # end-snippet-3

        # keep track of model input
        self.input = input


def test_mlp(learning_rate=0.01, L1_reg=0.001, L2_reg=0.001, n_epochs=2,
             dataset='C:/Users/Song/Research/biomed/trainingData.csv', n_hidden=2):
    train_set_x, train_set_t, t_column, observed, test_series = load_training_data(dataset)
    # compute number of minibatches for training, validation and testing
    input_shape = train_set_x.get_value(borrow=True).shape[1]
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    t = T.vector('t')  # the labels are presented as 1D vector of

    rng = numpy.random.RandomState(1234)

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        t=t,
        n_in=input_shape,
        n_hidden=n_hidden,
        n_out=1
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood()
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # start-snippet-5
    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [(T.grad(cost, param)) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]
    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        on_unused_input='ignore',
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x,
            t: train_set_t
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'

    epoch = 0
    while epoch < n_epochs + 1:
        epoch += 1
        avg_cost = train_model(epoch)
        print 'at epoch %d, cost is %f' % (epoch, avg_cost)

    params = [param.get_value() for param in classifier.params]
    test_array = numpy.asarray(test_series)[0]
    t = numpy.linspace(0, 600, 3000)
    p = numpy.asarray([])
    lin_in = numpy.dot(test_array, params[0][1:])
    for time in t:
        lin_in_all = time * params[0][0] + lin_in + params[1]
        lin_in_out = expit(lin_in_all)
        top_in = numpy.dot(lin_in_out, params[3]) + params[4]
        top_in_t = numpy.power(time * params[2], params[5])
        res = 1 / (1 + numpy.exp(top_in) * top_in_t)
        p = numpy.concatenate((p, res))
    # plot KM
    kmf.fit(t_column, event_observed=observed)
    kmf.survival_function_.plot(c='r')
    plt.plot(t, p, c='b')
    plt.show()

if __name__ == '__main__':
    test_mlp()
