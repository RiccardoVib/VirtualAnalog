import pickle
import random
import os
import tensorflow as tf
import numpy as np
from Code.Preprocess import my_scaler
import math

def get_data(data_dir, shuffle=False, w_length=0.01 ,seed=422):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #data_dir = '/Users/riccardosimionato/Datasets/VA/VA_results'
    #data_dir = 'C:/Users/riccarsi/Documents/GitHub/VA_pickle'
    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    meta = open(os.path.normpath('/'.join([data_dir, 'metadatas.pickle'])), 'rb')
    file_data = open(os.path.normpath('/'.join([data_dir, 'data.pickle'])), 'rb')
    Z = pickle.load(file_data)
    M = pickle.load(meta)
    inp = Z['inp']
    tar = Z['tar']
    ratios = M['ratio']
    threshold = M['threshold']
    fs = M['samplerate']
    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    inp = np.array(inp, dtype=np.float32)
    tar = np.array(tar, dtype=np.float32)
    ratios = np.array(ratios, dtype=np.float32)
    thresholds = np.array(threshold, dtype=np.float32)

    scaler = my_scaler()
    scaler_ratios = my_scaler()
    scaler_threshold = my_scaler()

    scaler_ratios.fit(ratios)
    scaler_threshold.fit(thresholds)
    scaler.fit(inp)
    inp = scaler.transform(inp)
    tar = scaler.transform(tar)
    thresholds = scaler_threshold.transform(thresholds)
    ratios = scaler_ratios.transform(ratios)

    zero_value_inp = (0 - scaler.min_data)/(scaler.max_data - scaler.min_data)
    zero_value_tar = (0 - scaler.min_data) / (scaler.max_data - scaler.min_data)
    zero_value_ratio = (0 - scaler_ratios.min_data) / (scaler_ratios.max_data - scaler_ratios.min_data)
    zero_value_threshold = (0 - scaler_threshold.min_data) / (scaler_threshold.max_data - scaler_threshold.min_data)
    zero_value = [zero_value_inp, zero_value_tar, zero_value_ratio, zero_value_threshold]
    scaler = [scaler, scaler_ratios, scaler_threshold]
    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------

    x, y, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    window = int(fs * w_length)
    all_inp, all_tar, r, thre = [], [],  [], []
    batch_size = 28
    for i in range(batch_size):
        for t in range(inp.shape[1]//window):

            inp_temp = np.array([inp[i, t * window:t * window + window], np.repeat(ratios[i], window)])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(tar[i, t*window:t*window + window])
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
    n_train = N//100*70
    n_val = (N-n_train)//2

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
    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x, y, x_val, y_val, x_test, y_test, scaler, zero_value