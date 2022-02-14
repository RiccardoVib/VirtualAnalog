import pickle
import random
import os
import tensorflow as tf
import numpy as np
from Code.Preprocess import my_scaler


def get_data_piano(data_dir, shuffle, w_length, seed=422):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    file_data = open(os.path.normpath('/'.join([data_dir, 'piano_data_sine.pickle'])), 'rb')
    Z = pickle.load(file_data)
    inp = Z['inp']
    tar = Z['tar']
    inp = np.array(inp, dtype=np.float32)
    tar = np.array(tar, dtype=np.float32)
    Z = [inp.T, tar.T]
    Z = np.array(Z)[:,:,0]

    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    scaler = my_scaler()
    scaler.fit(Z)

    inp = scaler.transform(inp)
    tar = scaler.transform(tar)
    zero_value = (0 - scaler.min_data) / (scaler.max_data - scaler.min_data)

    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------

    x, y, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    window = int(16000 * w_length)
    all_inp, all_tar = [], []

    for t in range(inp.shape[1] // window):
        inp_temp = np.array(inp[0, t * window:t * window + window])
        all_inp.append(inp_temp.T)
        tar_temp = np.array(tar[0, t * window:t * window + window])
        all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    w = 2  # n of column
    h = len(all_inp)  # n of row
    matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(h):
        matrix[i][0] = all_inp[i]
        matrix[i][1] = all_tar[i]
    if shuffle:
        np.random.shuffle(matrix)

    N = all_inp.shape[0]
    n_train = N // 100 * 70
    n_val = (N - n_train) // 2

    for n in range(n_train):
        x.append(matrix[n][0])
        y.append(matrix[n][1])

    for n in range(n_val):
        x_val.append(matrix[n_train + n][0])
        y_val.append(matrix[n_train + n][1])
        x_test.append(matrix[n_train + n_val + n][0])
        y_test.append(matrix[n_train + n_val + n][1])

    x = np.array(x)
    y = np.array(y)
    x_val = np.array(x_val)
    y_val = np.array(y_val)
    #x_test = np.array(x_test)
    #y_test = np.array(y_test)
    x_test = all_inp
    y_test = all_tar
    return x, y, x_val, y_val, x_test, y_test, scaler, zero_value