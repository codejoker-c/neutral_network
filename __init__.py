from utils.data_utils import get_CIFAR10_data
from classifiers.fc_net import *
from utils.solver import *
import matplotlib.pyplot as plt


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
    data = get_CIFAR10_data()
    for k, v in data.items():
        print("%s :" % k, v.shape)

    hidden_dim = 100
    num_classes = 10
    reg = 1e-5
    model = TwoLayerNet(hidden_dim=hidden_dim, num_classes=num_classes, reg=reg)
    optim_config = {"learning_rate": 1e-3, "num_echos": 10}
    solver = Solver(model, data, optim_config=optim_config)
    solver.train()
