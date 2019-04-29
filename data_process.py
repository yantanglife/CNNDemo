import numpy as np
import tensorflow.contrib.keras as kr
from sklearn.model_selection import train_test_split
from demo import get_all_datas, combine_data


def get_batch(x_train, y_train, batch_size=64):
    data_len = len(x_train)
    num_batch = int((data_len - 1) / batch_size) + 1
    # print(len(x_train), len(x_train[0]))
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x_train[indices]
    y_shuffle = y_train[indices]
    #print(y_train)
    #print(y_shuffle)
    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def get_train_valid_test_data(valid_size=0.2, test_size=0.2):
    x_acc_all_data, y_all_data = get_all_datas('./data', window_len=50)
    x_gyr_all_data, y_all_data = get_all_datas('./data', window_len=50, file_type='gyr')
    x_all_data = combine_data(x_acc_all_data, x_gyr_all_data)
    x_all_data = np.array(x_all_data)
    y_all_data = np.array(y_all_data)
    x_all_data = np.reshape(x_all_data, (-1, 50, 8))
    y_all_data = np.reshape(y_all_data, (-1))
    y_all_data = kr.utils.to_categorical(y_all_data, num_classes=10)
    #print(x_all_data.shape, y_all_data.shape)
    #print("y", y_all_data)
    x_train_valid, x_test, y_train_valid, y_test = train_test_split(
        x_all_data, y_all_data, test_size=test_size, random_state=41)
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train_valid, y_train_valid, test_size=valid_size * len(x_all_data) / float(len(x_train_valid)),
        random_state=42)
    return x_train, x_valid, x_test, y_train, y_valid, y_test
