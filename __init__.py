from KNN_test import KNN_test
from utils.data_utils import load_CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from linear_test import *


def output_img(X_train, y_train):
    classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    num_classes = len(classes)
    samples_per_class = 7
    for y, cls in enumerate(classes):
        idxs = np.flatnonzero(y_train == y)
        idxs = np.random.choice(idxs, samples_per_class, replace=False)
        for i, idx in enumerate(idxs):
            plt_idx = i * num_classes + y + 1
            plt.subplot(samples_per_class, num_classes, plt_idx)
            plt.imshow(X_train[idx].astype('uint8'))
            plt.axis('off')
            if i == 0:
                plt.title(cls)
    plt.show()


"""
分类器分类
"""
if __name__ == '__main__':
    cifar10_dir = 'datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # output_img(X_train,y_train)
    svm_test(X_train, y_train, X_test, y_test, learning_rate=1e-7, reg=2.5e4, num_iters=1500, verbose=True)
