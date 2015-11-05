__author__ = 'nelson'

from models.survivalNet import SurvivalNet
from utils.loadMatData import read_pickle
import theano
import theano.tensor as T
import numpy as np

theano.config.exception_verbosity='high'
def main():

    # Declare simbolic variables
    simb_x = T.matrix('x')  # the data is presented as rasterized images
    simb_o = T.ivector('o')  # observed death or not, 1 is death, 0 is right censored
    index = T.lscalar('index') # batch index

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

    # Define parameters
    solverArgs = {
        'iters': 100,
        'lr_rate': 0.001,
        'momentum': 0.9,
        'train_x': train_x,
        'train_y': train_y,
        'test_x': test_x,
        'test_y': test_y,
        'at_risk_x': at_risk_x,
        'batch_size': batch_size,
        'train_observed': train_observed,
        'n_hidden_layers': 8,
        'test_observed': test_observed,
        'input_shape': [243, 200],
        'corruption_level': 0.1,
        'n_train_batches': n_train_batches,
        'x': simb_x,
        'o': simb_o,
    }

    # Initialize model
    net = SurvivalNet(solverArgs)

    # Run training
    net.pretrain()
    net.train()
    return

if __name__ == '__main__':
    main()