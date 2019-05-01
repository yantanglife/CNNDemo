import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.metrics import classification_report
from time import time
from sklearn.svm import SVC
import matplotlib.pyplot as plt

from util.data_acquire import get_all_datas, combine_data


def split_to_train_test(x_, y_, size=0.8):
    x_train, x_test, y_train, y_test = [], [], [], []
    for x_data, y_data in zip(x_, y_):
        p = int(len(x_data) * size)
        x_train.append(x_data[: p])
        x_test.append(x_data[p:])
        y_train.append(y_data[: p])
        y_test.append(y_data[p:])
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def data_reshape(data, x_length=50, x_dim=6):
    x_train, x_test, y_train, y_test = data
    x_train = x_train.reshape((-1, x_length * x_dim))
    x_test = x_test.reshape((-1, x_length * x_dim))
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    return x_train, x_test, y_train, y_test


def svm_demo(data):
    x_train, x_test, y_train, y_test = data
    print("Train a SVM classification model")
    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)
    t0 = time()
    clf.fit(x_train, y_train)
    print("done in {:.3}s\nPredict:".format(time() - t0))
    # y_predict = clf.predict(x_test)
    score = clf.score(x_test, y_test)
    print("acc = {:.2%}".format(score))


def knn_demo(data):
    x_train, x_test, y_train, y_test = data
    print("Train a KNN classification model NO PCA")
    t0 = time()
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    print("done in {:.3}s\nPredict:".format(time() - t0))
    # y_predict = knn.predict(x_test)
    score = knn.score(x_test, y_test, sample_weight=None)
    print("acc = {:.2%}".format(score))


def pca_knn_demo(data):
    x_train, x_test, y_train, y_test = data
    print("Compute a PCA on the dataSet")
    t0 = time()
    pca = PCA(n_components=0.99)
    pca_fit = pca.fit(x_train)
    print("done in {:.3}s".format(time() - t0))
    print("Compute new train data")
    t0 = time()
    x_train_pca = pca_fit.transform(x_train)
    x_test_pca = pca_fit.transform(x_test)
    print("done in {:.3}s".format(time() - t0))
    print("x_train_pca.shape: ", x_train_pca.shape)
    print(pca.n_components_)
    # Train a KNN classification model
    print("Train a KNN classification model")
    t0 = time()
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train_pca, y_train)
    print("done in {:.3}s".format(time() - t0))
    print("Predict:")
    # y_predict = knn.predict(x_test_pca)
    score = knn.score(x_test_pca, y_test, sample_weight=None)
    print("acc = {:.2%}".format(score))


def show_data_demo(data, data_dim='x', data_num=30, title=""):
    """
    just show 3 classes' data.
    :param data: X_acc_all_data or X_gyr_all_data
    :param data_dim: which dimension to show.
    :param data_num: this represent the num of data group.Here, length are (window_len x data_num).
    :param title:
    :return:
    """
    if data_dim not in ['x', 'y', 'z']:
        print("error")
        return
    dim = 0 if data_dim == 'x' else (1 if data_dim == 'y' else 2)
    # fig, ax = plt.subplots(nrows=2, ncols=1)
    x1, x2, x3 = [], [], []
    for i in range(data_num):
        for j, k, l in zip(data[0][i], data[1][i], data[2][i]):
            x1.append(j[dim])
            x2.append(k[dim])
            x3.append(l[dim])
    plt.plot(x1, c='r')
    plt.plot(x2, c='g')
    plt.plot(x3)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    X_acc_all_data, Y_all_data = get_all_datas('./data', window_len=50, step=20)
    X_gyr_all_data, Y_all_data = get_all_datas('./data', window_len=50, step=20, file_type='gyr')

    X_all_data = combine_data(X_acc_all_data, X_gyr_all_data)
    all_data = split_to_train_test(X_all_data, Y_all_data)
    # show_data_demo(data=X_acc_all_data, title="test")
    all_data = data_reshape(all_data)
    print('-' * 50)
    pca_knn_demo(all_data)
    print('-' * 50)
    svm_demo(all_data)
    print('-' * 50)
    knn_demo(all_data)
    print('-' * 50)

