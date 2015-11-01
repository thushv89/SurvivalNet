__author__ = 'nelson'

from models.survivalNet import SurvivalNet
from models.stackedAutoEncoders import StackedAutoEncoders
from utils.loadMatData import read_pickle
import theano
import numpy as np

#theano.config.exception_verbosity='high'
def main():

    # Read data
    observed, x, survival_time, at_risk_x = read_pickle()

    # Split data
    test_size = len(x) / 3
    train_x = x[test_size:]
    train_y = survival_time[test_size:]
    train_observed = observed[test_size:]
    test_observed = observed[:test_size]
    test_x = x[:test_size]
    test_y = survival_time[:test_size]
    batch_size = len(train_x)
    n_train_batches = len(train_x)/batch_size

    #########################
    # PRETRAINING THE MODEL #
    #########################
    # Define parameters
    #preSolverArgs = {
    #    'iters': 40,
    #    'lr_rate': 1.0,
    #    'W': None,
    #    'b': None,
    #    'train_x': train_x,
    #    'at_risk_x': at_risk_x,
    #    'batch_size': batch_size,
    #    'n_out': 200,
    #    'n_layers': 8,
    #    'n_train_batches': n_train_batches,
    #    'corruption_level': 0.1,
    #    'numpy_rng': np.random.RandomState(89677),
    #    'input_shape': [243, 200]
    #}
    #preNet = StackedAutoEncoders(preSolverArgs)
    #preNet.train()

    ########################
    # FINETUNING THE MODEL #
    ########################
    #print "Finetuning Model"
    # Define parameters
    solverArgs = {
        'iters': 100,
        'lr_rate': 0.0001,
        'momentum': 0.9,
        'W': None,
        'b': None,
        'train_x': train_x,
        'train_y': train_y,
        'test_x': train_x,
        'test_y': train_y,
        'at_risk_x': at_risk_x,
        'batch_size': batch_size,
        'train_observed': train_observed,
        'n_hidden_layers': 1,
        'test_observed': train_observed,
        'input_shape': [243, 200]
        #'input_data': preNet.layers[-1].input_data
    }

    # Initialize model
    net = SurvivalNet(solverArgs)

    # Run training
    net.train()
    return

if __name__ == '__main__':
    main()