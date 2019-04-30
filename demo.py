import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from time import time
from scipy import signal
from sklearn.svm import SVC
import matplotlib.pyplot as plt


def distance(x, y, z):
    return pow(x*x + y*y + z*z, 0.5)


def get_data(file_name, window_len, step):
    data, all_datas = [], []
    with open(file_name, 'r') as file:
        all_lines = file.readlines()
        for idx, line in enumerate(all_lines, 1):
            _, x, y, z = [float(i) for i in line.split()]
            # data.append([x, y, z, distance(x, y, z)])
            data.append([x, y, z])
            '''
            if idx % window_len == 0: 
                all_datas.append(data)
                data = []
            '''
    # print(data[:10])
    #b, a = signal.butter(8, 0.2, 'lowpass')
    #filted_data = signal.filtfilt(b, a, data, axis=0)

    filted_data = data
    # print(filted_data[:10])
    '''
    filted_data_1 = signal.filtfilt(b, a, data, axis=1)
    filted_data_2 = signal.filtfilt(b, a, data, axis=2)
    filted_data = [[data_0, data_1, data_2] for data_0, data_1, data_2 in
                   zip(filted_data_0, filted_data_1, filted_data_2)]
    '''
    for idx in range(0, len(filted_data), step):
        if idx + window_len < len(filted_data):
            all_datas.append(filted_data[idx: idx + window_len])
    return all_datas
    '''
    for i in range(0, len(datas), window_len):
        yield datas[i: i + window_len]
    '''


def get_all_datas(dir_path, window_len, step=30, file_type='acc'):
    x_all_data, y_all_data = [], []
    file_name_list = os.listdir(dir_path)
    sorted(file_name_list)
    for idx, file_name in enumerate(file_name_list, 0):
        if idx >= 10:
            label = idx - 10
        else:
            label = idx
        if file_name.startswith(file_type):
            data = get_data(os.path.join(dir_path, file_name), window_len, step=step)
            x_all_data.append(data)
            y_all_data.append([label for _ in range(len(data))])
    return x_all_data, y_all_data


def combine_data(acc_all_data, gry_all_data):
    comb_data = []
    for acc_data, gry_data in zip(acc_all_data, gry_all_data):
        temp = np.concatenate((np.array(acc_data), np.array(gry_data)), axis=2)
        # print(len(acc_data), len(acc_data[0]))
        # print(temp.shape)
        comb_data.append(temp)
    return comb_data


def split_to_train_test(x_all_data, y_all_data, size=0.8):
    x_train, x_test, y_train, y_test = [], [], [], []
    for x_data, y_data in zip(x_all_data, y_all_data):
        p = int(len(x_data) * size)
        x_train.append(x_data[: p])
        x_test.append(x_data[p:])
        y_train.append(y_data[: p])
        y_test.append(y_data[p:])
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)


def svm_demo(all_data):
    x_train, x_test, y_train, y_test = all_data

    x_train = x_train.reshape((-1, 50 * 6))
    x_test = x_test.reshape((-1, 50 * 6))
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
              decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
              max_iter=-1, probability=False, random_state=None, shrinking=True,
              tol=0.001, verbose=False)

    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    print("acc = {:.2%}".format(clf.score(x_test, y_test)))


def knn_demo(all_data):
    x_train, x_test, y_train, y_test = all_data

    x_train = x_train.reshape((-1, 50 * 6))
    x_test = x_test.reshape((-1, 50 * 6))
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    print("Train a KNN classification model NO PCA")
    t0 = time()
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(x_train, y_train)
    print("done in {:.3}s".format(time() - t0))
    print("Predict:")
    y_predict = knn.predict(x_test)
    score = knn.score(x_test, y_test, sample_weight=None)
    print("acc = {:.2%}".format(score))

def pca_knn_demo(all_data):
    x_train, x_test, y_train, y_test = all_data
    # all faces
    print(x_train.shape)
    print(x_test.shape)

    x_train = x_train.reshape((-1, 50 * 6))
    x_test = x_test.reshape((-1, 50 * 6))
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    print(y_train.shape)
    print(y_test.shape)
    # compute a PCA on the face dateSet
    print("Compute a PCA on the dataSet")
    t0 = time()
    pca = PCA(n_components=0.99)
    pca_fit = pca.fit(x_train)
    print("done in {:.3}s".format(time() - t0))

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
    y_predict = knn.predict(x_test_pca)
    score = knn.score(x_test_pca, y_test, sample_weight=None)
    print("acc = {:.2%}".format(score))


if __name__ == "__main__":
    # data = get_data("./data/accData0.txt", 100)
    # print(len(data))
    # a = np.array(data)
    x_acc_all_data, y_all_data = get_all_datas('./data', window_len=50)
    fig, ax = plt.subplots(nrows=2, ncols=1)
    x1, x2, x3 = [], [], []
    y1, y2, y3 = [], [], []
    for i in range(30):
        for j, k, l in zip(x_acc_all_data[0][i], x_acc_all_data[1][i], x_acc_all_data[2][i]):
            x1.append(j[0])
            x2.append(k[0])
            x3.append(l[0])
            y1.append(j[1])
            y2.append(k[1])
            y3.append(l[1])
    ax[0].plot(x1, c='r')
    ax[0].plot(x2, c='g')
    ax[0].plot(x3)
    ax[1].plot(y1, c='r')
    ax[1].plot(y2, c='g')
    ax[1].plot(y3)
    plt.title('8-0.3')
    plt.show()

    x_gyr_all_data, y_all_data = get_all_datas('./data', window_len=50, file_type='gyr')
    fig, ax = plt.subplots(nrows=2, ncols=1)
    x1, x2, x3 = [], [], []
    y1, y2, y3 = [], [], []
    for i in range(30):
        for j, k, l in zip(x_gyr_all_data[0][i], x_gyr_all_data[1][i], x_gyr_all_data[2][i]):
            x1.append(j[0])
            x2.append(k[0])
            x3.append(l[0])
            y1.append(j[1])
            y2.append(k[1])
            y3.append(l[1])
    ax[0].plot(x1, c='r')
    ax[0].plot(x2, c='g')
    ax[0].plot(x3)
    ax[1].plot(y1, c='r')
    ax[1].plot(y2, c='g')
    ax[1].plot(y3)
    plt.title('gyr-8-0.3')
    plt.show()
    x_all_data = combine_data(x_acc_all_data, x_gyr_all_data)
    # print(x_all_data[0][0]
    all = split_to_train_test(x_all_data, y_all_data)
    pca_knn_demo(all)
    svm_demo(all)
    knn_demo(all)
