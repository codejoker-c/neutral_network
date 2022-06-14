import numpy as np


def softmax_loss(W, X, y, reg):
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # forward pass
    num_train = X.shape[0]

    scores = X.dot(W)
    exp_scores = np.exp(scores)
    sum_escores = np.sum(exp_scores, axis=1)
    y_escores = np.zeros(num_train)
    y_escores += exp_scores[range(num_train), y]
    cd_sescores = y_escores / sum_escores
    ln_scores = np.log(cd_sescores)
    loss += -np.sum(ln_scores) / num_train
    loss += reg * np.sum(W * W)

    # backpropgation
    num_cls = W.shape[1]
    dcd_sescores = -1 / (cd_sescores)
    temp1 = np.zeros((num_train, num_cls))
    sum_s_2 = np.power(sum_escores, 2)
    temp2 = -y_escores / sum_s_2
    temp1 += temp2.reshape(num_train, 1)
    temp1[range(num_train), y] = (sum_escores - y_escores) / sum_s_2
    dexp_scores = dcd_sescores.reshape(num_train, 1) * temp1
    dscores = dexp_scores * exp_scores
    dW += np.dot(X.T, dscores) / num_train
    dW += 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
