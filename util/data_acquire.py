import os
import numpy as np
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
    #b, a = signal.butter(8, 0.2, 'lowpass')
    #filted_data = signal.filtfilt(b, a, data, axis=0)

    filted_data = data

    for idx in range(0, len(filted_data), step):
        if idx + window_len < len(filted_data):
            all_datas.append(filted_data[idx: idx + window_len])
    return all_datas


def get_all_datas(dir_path, window_len, step, file_type='acc'):
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


if __name__ == "__main__":
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
    all = split_to_train_test(x_all_data, y_all_data)
