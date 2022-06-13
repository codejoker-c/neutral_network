import numpy as np


class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        """
        :param X: (N,size(3072)) 训练集
        :param y: (N,1)    训练集label
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1):
        """
        :param X: (k,size(3072)) 测试集
        :param k: 选取k个最近的，然后选择k个中label最多的作为预测label
        :return:  预测label y_pred
        """
        num_test = X.shape[0]

        dists = self.compute_distance(X)
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            idxs = np.argsort(dists[i])
            closest_y = [self.y_train[j] for j in idxs[:k]]
            y_pred[i] = max(closest_y, key=closest_y.count)

        return y_pred

    def compute_distance(self, X):
        """
        :param X: 测试集
        :return: 与训练集之间的欧式距离矩阵 dists
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]

        # 使用欧式距离，将式子拆分计算
        temp1 = X.dot(self.X_train.T)
        temp2 = np.sum(self.X_train * self.X_train, axis=1).reshape((1, num_train))
        temp3 = np.sum(X * X, axis=1).reshape((num_test, 1))

        dists = np.zeros((num_test, num_train))
        dists[:, :] = np.sqrt(temp2 + temp3 - 2 * temp1)
        return dists
