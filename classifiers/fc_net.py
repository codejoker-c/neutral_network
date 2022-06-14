from builtins import object
import numpy as np

from utils.layers_utils import *
from utils.layers import *

class TwoLayerNet(object):


    def __init__(
            self,
            input_dim=3 * 32 * 32,
            hidden_dim=100,
            num_classes=10,
            weight_scale=1e-3,
            reg=0.0,
    ):
        self.params = {}
        self.reg = reg

        self.params['W1'] = np.random.randn(input_dim, hidden_dim) * weight_scale
        self.params['W2'] = np.random.randn(hidden_dim, num_classes) * weight_scale
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['b2'] = np.zeros(num_classes)


    def loss(self, X, y=None):

        scores = None

        out1, cache1 = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        scores, cache2 = affine_forward(out1, self.params['W2'], self.params['b2'])

        if y is None:
            return scores

        loss, grads = 0, {}
        loss, dscores = softmax_loss(scores, y)
        loss += 0.5 * self.reg * (
                    np.sum(self.params['W2'] * self.params['W2']) + np.sum(self.params['W1'] * self.params['W1']))
        dout1, dW2, db2 = affine_backward(dscores, cache2)
        dX, dW1, db1 = affine_relu_backward(dout1, cache1)

        grads['W2'] = dW2 + 0.5 * self.reg * 2 * self.params['W2']
        grads['b2'] = db2
        grads['W1'] = dW1 + 0.5 * self.reg * 2 * self.params['W1']
        grads['b1'] = db1

        return loss, grads

