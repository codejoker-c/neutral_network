import numpy as np
from .linear_svm import svm_loss
from .softmax import softmax_loss


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(
            self,
            X,
            y,
            learning_rate=1e-3,
            reg=1e-5,
            num_iters=100,
            batch_size=200,
            verbose=False
    ):
        """
        :param X: (N,size+1) 训练集
        :param y: (N,1)
        :param learning_rate: 学习率
        :param reg: 规格化强度
        :param num_iters: 优化过程中迭代的次数
        :param batch_size: 每次优化选取的训练集样本数
        :param verbose: 是否打印训练过程，true则打印
        :return:
        """
        num_train, dim = X.shape
        num_cls = np.max(y) + 1
        # W 初始化
        if self.W is None:
            self.W = 0.001 * np.random.randn(dim, num_cls)

        loss_history = []
        # 训练过程
        for it in range(num_iters):
            index_arr = np.random.choice(range(num_train), size=batch_size)
            X_batch = X[index_arr, :]
            y_batch = y[index_arr]

            # 随机梯度下降
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W += -learning_rate * grad

            if verbose and it % 100 == 0:
                print("iteration %d / %d: loss %f" % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
        :param X: (num_test,size+1)
        :return: y_pred
        """
        num_test = X.shape[0]
        y_pred = np.zeros(num_test)
        scores = np.dot(X, self.W)
        y_pred += np.argmax(scores, axis=1)

        return y_pred

    def loss(self, X_batch, y_batch, reg):
        pass


class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return svm_loss(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return softmax_loss(self.W, X_batch, y_batch, reg)
