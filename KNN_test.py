from classifiers import KNearestNeighbor
import numpy as np

def KNN_test(X_train, y_train, X_test, y_test,k=1):
    # 对数据取样
    num_training = 5000
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]

    num_test = 500
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 将数据拉平
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    print(X_train.shape)
    print(X_test.shape)
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)
    # 展示欧式距离图像
    # dists = classifier.compute_distance(X_test)
    # plt.imshow(dists, interpolation='none')
    # plt.show()
    y_test_pre = classifier.predict(X_test, k)
    corrects = np.sum(y_test == y_test_pre)
    accuracy = corrects / y_test.shape[0]
    print('accuracy: %f' % (accuracy))