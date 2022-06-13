import numpy as np


def softmax_loss(W, X, y, reg):
    loss = 0.0
    grad_W = np.zeros(W.shape)

    # 前向传递计算 loss
    num_train = X.shape[0]
    scores = np.dot(X, grad_W)
    exp_scores = np.exp(scores)
    sum_escores = np.sum(exp_scores, axis=1)
    y_escores = np.zeros(num_train)
    y_escores += exp_scores[range(num_train), y]
    cd_scores = y_escores / sum_escores
    ln_scores = -np.log(cd_scores)

    loss += np.sum(ln_scores) / num_train
    loss += reg * np.sum(W * W)

    # 反向传播计算 grad_W
    num_cls = W.shape[1]
    grad_cd_scores = -1 / cd_scores
    temp1 = np.zeros((num_train, num_cls))
    sum_s_2 = np.power(sum_escores, 2)
    temp2 = -y_escores / sum_s_2
    temp1 += temp2.reshape(num_train, 1)
    temp1[range(num_train), y] = (sum_escores - y_escores) / sum_s_2
    grad_exp_scores = grad_cd_scores.reshape(num_train, 1) * temp1
    grad_scores = grad_exp_scores * exp_scores

    grad_W += np.dot(X.T, grad_scores) / num_train
    grad_W += 2 * reg * W

    return loss, grad_W
