from classifiers import LinearSVM, Softmax
import numpy as np
import matplotlib.pyplot as plt
from utils.gradient_check import grad_check_sparse
from classifiers.linear_svm import svm_loss
from classifiers.softmax import softmax_loss


def svm_test(X_train, y_train, X_test, y_test, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200,
             verbose=False):
    """
    数据预处理
    """
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500

    # 核验集
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    # 训练集
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    #
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # 测试集
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 拉平
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # 求出测试集中49000张图片的均值，(3072,)
    mean_image = np.mean(X_train, axis=0)
    # print(mean_image[:10])
    # plt.figure(figsize=(4, 4))
    # plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))
    # plt.show()

    # 减去平均值，增强算法稳定性
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    # 增加一列1，这样b就可以写在W中，就不用再考虑单独优化b了
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)
    # print(X_dev.shape)

    # """
    # check gradient
    # """
    # W = np.random.randn(3073, 10) * 0.0001
    # loss, grad = svm_loss(W, X_dev, y_dev, 0.0)
    #
    # f = lambda w: svm_loss(w, X_dev, y_dev, 0.0)[0]  # 根据权值计算loss的lambda表达式
    # grad_numerical = grad_check_sparse(f, W, grad)
    #
    # loss, grad = svm_loss(W, X_dev, y_dev, 5e1)
    # f = lambda w: svm_loss(w, X_dev, y_dev, 5e1)[0]
    # grad_numerical = grad_check_sparse(f, W, grad)

    """
    分类
    """
    svm = LinearSVM()
    loss_history = svm.train(X_train, y_train, learning_rate, reg, num_iters, batch_size, verbose)

    # 画出loss变化
    plt.plot(loss_history)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.show()

    y_train_pred = svm.predict(X_train)
    print('training accuracy: %f' % (np.mean(y_train == y_train_pred),))
    y_val_pred = svm.predict(X_val)
    print('validation accuracy: %f' % (np.mean(y_val == y_val_pred),))

    y_test_pred = svm.predict(X_test)
    print('test accuracy: %f' % (np.mean(y_test == y_test_pred)))


def softmax_test(X_train, y_train, X_test, y_test, learning_rate=1e-3, reg=1e-5, num_iters=100, batch_size=200,
                 verbose=False):
    """
    数据预处理
    """
    num_training = 49000
    num_validation = 1000
    num_test = 1000
    num_dev = 500

    # 核验集
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]

    # 训练集
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]

    #
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # 测试集
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # 拉平
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1))

    # 求出测试集中49000张图片的均值，(3072,)
    mean_image = np.mean(X_train, axis=0)
    # print(mean_image[:10])
    # plt.figure(figsize=(4, 4))
    # plt.imshow(mean_image.reshape((32, 32, 3)).astype('uint8'))
    # plt.show()

    # 减去平均值，增强算法稳定性
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    X_dev -= mean_image

    # 增加一列1，这样b就可以写在W中，就不用再考虑单独优化b了
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)
    # print(X_dev.shape)

    """
    check gradient
    """
    W = np.random.randn(3073, 10) * 0.0001
    loss, grad = softmax_loss(W, X_dev, y_dev, 0.0)

    f = lambda w: svm_loss(w, X_dev, y_dev, 0.0)[0]  # 根据权值计算loss的lambda表达式
    grad_numerical = grad_check_sparse(f, W, grad)

    loss, grad = softmax_loss(W, X_dev, y_dev, 5e1)
    f = lambda w: softmax_loss(w, X_dev, y_dev, 5e1)[0]
    grad_numerical = grad_check_sparse(f, W, grad)

    # """
    # 分类
    # """
    # softmax = Softmax()
    # loss_history = softmax.train(X_train, y_train, learning_rate, reg, num_iters, batch_size, verbose)
    #
    # # 画出loss变化
    # plt.plot(loss_history)
    # plt.xlabel('Iteration number')
    # plt.ylabel('Loss value')
    # plt.show()
    #
    # y_train_pred = softmax.predict(X_train)
    # print('training accuracy: %f' % (np.mean(y_train == y_train_pred),))
    # y_val_pred = softmax.predict(X_val)
    # print('validation accuracy: %f' % (np.mean(y_val == y_val_pred),))
