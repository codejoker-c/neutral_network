from builtins import range
import numpy as np


def affine_forward(x, w, b):
    out = None
    N = x.shape[0]

    x1 = np.reshape(x, (N, -1))
    out = x1.dot(w) + b

    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    x, w, b = cache
    dx, dw, db = None, None, None
    x1 = np.reshape(x, (x.shape[0], -1))

    db = np.sum(dout, axis=0)
    dw = np.dot(x1.T, dout)
    dx = np.reshape(np.dot(dout, w.T), (x.shape))

    return dx, dw, db


def relu_forward(x):
    out = None

    out = np.zeros(x.shape)
    out += x
    out[out < 0] = 0

    cache = x
    return out, cache


def relu_backward(dout, cache):
    dx, x = None, cache

    dx = np.zeros(x.shape)
    dx = np.add(dx, dout)
    dx[x < 0] = 0

    return dx


def svm_loss(x, y):
    N = x.shape[0]
    C = x.shape[1]
    loss, dx = 0.0, None

    # forword pass
    loss_matrix = x - x[range(N), y].reshape(N, 1) + 1
    loss_matrix[loss_matrix <= 0] = 0
    loss_arr = np.sum(loss_matrix, axis=1) - 1
    loss += np.sum(loss_arr) / N

    # backpropagation
    dloss_matrix = np.ones((N, C)) / N
    dloss_matrix[loss_matrix == 0] = 0
    dx = np.zeros((N, C))
    dx += dloss_matrix
    dx[range(N), y] -= np.sum(dloss_matrix, axis=1)

    return loss, dx


def softmax_loss(x, y):
    loss, dx = None, None

    num_train = x.shape[0]
    scores = x - np.max(x, axis=1).reshape(num_train, 1)
    normalized_scores = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(num_train, 1)
    loss = -np.sum(np.log(normalized_scores[np.arange(num_train), y]))
    loss /= num_train

    normalized_scores[np.arange(num_train), y] -= 1
    dx = normalized_scores / num_train

    return loss, dx
