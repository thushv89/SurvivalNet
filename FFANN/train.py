__author__ = 'Song'
from loadData import load_training_data
from loadMatData import load_data
import matplotlib.pyplot as plt
from scipy.special import expit
from path import Path
from lifelines.utils import _naive_concordance_index
import theano.tensor as T
import numpy
import theano
from mlp import MLP
# from lifelines import KaplanMeierFitter
# kmf = KaplanMeierFitter()

VA = Path('data/VA.mat')
LUAD_P = Path('data/LUAD_P.mat')
LUSC_P = Path('data/LUSC_P.mat')
Brain_P = Path('data/Brain_P.mat')
AML = 'C:/Users/Song/Research/biomed/Survival/trainingData.csv'


def train(x_train,  o_train, t_val, o_val, x_val, t_test, o_test, x_test, learning_rate=0.00001, L1_reg=0.000,
          L2_reg=0.075, n_epochs=300, n_hidden=12, testing=False):

    # compute number of minibatches for training, validation and testing
    input_shape = x_train.shape[1]
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'

    rng = numpy.random.RandomState(1234)
    x = T.matrix('x')
    o = T.vectors('o')

    # construct the MLP class
    classifier = MLP(
        rng=rng,
        input=x,
        n_in=input_shape,
        n_hidden=n_hidden,
        n_out=1
    )

    # start-snippet-4
    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(o)
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
    index = T.iscalar()
    train_model = theano.function(
        on_unused_input='ignore',
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: x_train,
            o: o_train
        }
    )

    val_fn = theano.function(
        on_unused_input='ignore',
        inputs=[index],
        outputs=classifier.outputLayer.hazard_ratio,
        givens={
            x: x_val,
            o: o_val
        }
    )

    test_fn = theano.function(
        on_unused_input='ignore',
        inputs=[index],
        outputs=classifier.outputLayer.hazard_ratio,
        givens={
            x: x_test,
            o: o_test
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    test_c_index = 0
    best_val = 0
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        avg_cost = train_model(epoch)
        # learning_rate *= 0.95
        val_hazard_rate = val_fn(epoch)
        val_c_index = _naive_concordance_index(t_val, -val_hazard_rate, o_val)
        if val_c_index >= best_val:
            best_val = val_c_index
            test_hazard = test_fn(epoch)
            test_c_index = _naive_concordance_index(t_test, -test_hazard, o_test)
        print 'at epoch %d, cost is %f, validate c_index is %f' % (epoch, avg_cost, val_c_index)

    print 'best validate score is', best_val
    if testing:
        print 'final test score', test_c_index
        return best_val, test_c_index
    return best_val
