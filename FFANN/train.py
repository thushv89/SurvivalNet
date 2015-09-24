__author__ = 'Song'
from loadData import load_training_data
import matplotlib.pyplot as plt
from scipy.special import expit
from lifelines import KaplanMeierFitter
from lifelines.utils import _naive_concordance_index
kmf = KaplanMeierFitter()
import theano.tensor as T
import numpy
import theano
from mlp import MLP


def main(learning_rate=0.0000025, L1_reg=0.5, L2_reg=0.75, n_epochs=300,
             dataset='C:/Users/Song/Research/biomed/Survival/trainingData.csv', n_hidden=100):
    train_set_x,  discrete_observed, survival_time, observed, avg_series, test_data = load_training_data(dataset)
    # compute number of minibatches for training, validation and testing
    input_shape = train_set_x.shape[1]
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
            x: train_set_x,
            o: discrete_observed
        }
    )
    # end-snippet-5

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    c = []
    epoch = 0
    while epoch < n_epochs:
        epoch += 1
        avg_cost = train_model(epoch)
        learning_rate *= 0.95
        params = [param.get_value() for param in classifier.params]
        c_index = get_c_index(params, test_data, survival_time, observed)
        c.append(c_index)
        print 'at epoch %d, cost is %f, test c_index is %f' % (epoch, avg_cost, c_index)
    plt.ylim(0.2, 0.8)
    plt.plot(range(len(c)), c, c='r', marker='o', lw=5, ms=10, mfc='c')
    plt.show()
    # t = numpy.linspace(0, 600, 2401)
    # p = []
    # for time in t:
    #     test_x = numpy.insert(test_series, 0, time)
    #     temp = expit(numpy.dot(test_x, params[0]) + params[1])
    #     prob = expit(numpy.dot(temp, params[2]) + params[3])
    #     p.append(prob)
    # y = []
    # for i in xrange(len(t)):
    #     j = i
    #     y_pred = 1
    #     while j > 0:
    #         y_pred *= 1 - p[j]
    #         j -= 1
    #     y.append(y_pred)

    # # plot KM
    # kmf.fit(survival_time, event_observed=observed)
    # kmf.survival_function_.plot(c='r')
    # plt.plot(t, y, c='g')
    # plt.show()


def get_c_index(params, test_data, survival_time, observed):
    hazard_rate = []
    for data in test_data:
        temp = expit(numpy.dot(data, params[0]) + params[1])
        hazard = expit(numpy.dot(temp, params[2]) + params[3])
        hazard_rate.append(hazard)

    return _naive_concordance_index(survival_time, hazard_rate, observed)

if __name__ == '__main__':
    main()
