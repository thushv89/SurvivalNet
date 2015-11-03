__author__ = 'nelson'
import theano.tensor as T

def ReLU(X, _):
    return T.maximum(X, 0.)

# Alpha >= 1
def leakyReLU(X, alpha):
    return T.maximum(X, X/alpha)

def RReLU():
    return

def PReLU(X, alpha):
    return



