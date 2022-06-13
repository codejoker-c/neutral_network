import numpy as np


def svm_loss(W, X, y, reg):
    """
    :param W: (size+1)*cls
    :param X: N*(size+1) 训练集
    :param y: N*1 训练集label
    :param reg: 超参数，用于regularization
    :return: 返回loss 和 dW
    """
    loss = 0.0
    dW = np.zeros(W.shape)
    num_train = X.shape[0]
    num_cls = W.shape[1]

    # 前向传递 计算loss
    scores = X.dot(W)
    loss_matrix = scores - scores[range(num_train), y].reshape(num_train, 1) + 1
    loss_matrix[loss_matrix <= 0] = 0
    loss_arr = np.sum(loss_matrix, axis=1) - 1
    loss += np.sum(loss_arr) / num_train
    loss += reg * np.sum(W * W)

    # 反向传播 计算dW
    dW += 2 * reg * W
    dloss_matrix = np.ones((num_train, num_cls)) / num_train
    dloss_matrix[loss_matrix == 0] = 0
    dscores = np.zeros((num_train, num_cls))
    dscores += dloss_matrix
    dscores[range(num_train), y] -= np.sum(dloss_matrix, axis=1)
    dW += np.dot(X.T, dscores)
    return loss, dW

