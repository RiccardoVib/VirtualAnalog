import pickle
import random
import os
import tensorflow as tf
import numpy as np
from Code.Preprocess import my_scaler_stft

def get_data(data_dir, seed=422):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    file_index = open(os.path.normpath('/'.join([data_dir, 'index_data.pickle'])), 'rb')
    # file_lenghts = open('lenghts_data.pickle', 'rb')
    file_data = open(os.path.normpath('/'.join([data_dir, 'data.pickle'])), 'rb')
    indeces = pickle.load(file_index)
    Z = pickle.load(file_data)
    # lenghts = pickle.load(file_lenghts)

    name = Z['name']
    Z = Z['data']
    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    Z = np.array(Z)

    # Z = Z[np.concatenate([np.squeeze(np.array([[101], [100], [98]])), np.array([90, 97, 99])])]
    # input = np.array([[0], [1], [2]])
    # target = np.array([3, 4, 5])

    scaler = my_scaler_stft()
    scaler.fit(Z)
    Z = scaler.transform(Z)
    zero_value = (0 - scaler.min_data)/(scaler.max_data - scaler.min_data)

    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------

    input = indeces['input_data']
    target = indeces['target_data']


    # to debug
    # input = input[:14]
    # target = target[:14]

    w = 2  # n of column
    h = len(target)  # n of row
    matrix = [[0 for x in range(w)] for y in range(h)]

    for i in range(h):
        matrix[i][0] = input[i]
        matrix[i][1] = target[i]

    np.random.shuffle(matrix)

    x = []  # input
    y = []  # target
    x_val, y_val, x_test, y_test = [], [], [], []

    # lenght of train set = 70%, Val=15%, Test=15%
    n_train = len(input) - len(input) // 7
    n_val_test = len(input) // 14

    for i in range(n_train):
        x.append(matrix[i][0])
        y.append(matrix[i][1])

    for i in range(n_val_test):
        x_val.append(matrix[n_train + i][0])
        y_val.append(matrix[n_train + i][1])
        x_test.append(matrix[n_train + n_val_test + i][0])
        y_test.append(matrix[n_train + n_val_test + i][1])

    x = np.array(x)
    y = np.array(y)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x, y, x_val, y_val, x_test, y_test, Z, scaler, zero_value, indeces, name
