import pickle
import random
import os
import tensorflow as tf
import numpy as np
from Code.Preprocess import my_scaler
import math

def get_data(data_dir, batch_size=28, seed=422):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    #data_dir = '/Users/riccardosimionato/Datasets/VA/VA_results'
    data_dir = 'C:/Users/riccarsi/Documents/GitHub/VA_pickle'
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
    fs = M['samplerate']
    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    inp = np.array(inp, dtype=np.float32)
    tar = np.array(tar, dtype=np.float32)
    ratios = np.array(ratios, dtype=np.float32)

    scaler_ratios = my_scaler()
    scaler_inp = my_scaler()
    scaler_tar = my_scaler()
    scaler_ratios.fit(ratios)
    scaler_inp.fit(inp)
    scaler_tar.fit(tar)
    inp = scaler_inp.transform(inp)
    tar = scaler_tar.transform(tar)
    ratios = scaler_ratios.transform(ratios)

    zero_value_inp = (0 - scaler_inp.min_data)/(scaler_inp.max_data - scaler_inp.min_data)
    zero_value_tar = (0 - scaler_tar.min_data) / (scaler_tar.max_data - scaler_tar.min_data)
    zero_value_ratio = (0 - scaler_ratios.min_data) / (scaler_ratios.max_data - scaler_ratios.min_data)
    zero_value = [zero_value_inp, zero_value_tar, zero_value_ratio]
    scaler = [scaler_inp, scaler_tar, scaler_ratios]
    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------
    #N = len(inp[0]) #32097856
    #n_train = N//100*70
    #n_val = N//100*15
    x, y, x_val, y_val, x_test, y_test, r = [], [], [], [], [], [], []
    #batch_size = 1
    #inp = inp[0:batch_size, :]
    #tar = tar[0:batch_size, :]
    # for i in range(batch_size):
    #     x.append(inp[i, 0:n_train])
    #     x_val.append(inp[i, n_train:n_train+n_val])
    #     x_test.append(inp[i, n_train+n_val:])
    #     r.append(ratios[i])
    # for t in range(batch_size):
    #     y.append(tar[t, 0:n_train])
    #     y_val.append(tar[t, n_train:n_train+n_val])
    #     y_test.append(tar[t, n_train+n_val:])
    #
    # 100 ms = 0.1*fs
    window = int(fs * 0.2 / 6)
    all_inp, all_tar = [], []
    for i in range(batch_size):
        #for t in range(inp.shape[1]-window):
        for t in range(10):
            all_inp.append(inp[i, t:t + window])
            all_tar.append(tar[i, t:t + window])
            r.append(ratios[i])

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)
    r = np.array(r)

    N = all_inp.shape[0]
    n_train = N//100*70
    n_val = (N-n_train)//2

    x = all_inp[:n_train]
    y = all_tar[:n_train]
    r_train = r[:n_train]
    x_val = all_inp[n_train:n_train+n_val]
    y_val = all_tar[n_train:n_train+n_val]
    r_val = r[n_train:n_train+n_val]
    x_test = all_inp[n_train+n_val:]
    y_test = all_tar[n_train+n_val:]
    r_test = r[n_train+n_val:]

    return x, y, x_val, y_val, x_test, y_test, r_train, r_val, r_test, scaler, zero_value


    # n_row_train = int(n_train/window)
    # new_N_train = int(n_row_train*window)
    #
    # n_row_val = int(n_val/window)
    # new_N_val = int(n_row_val * window)
    #
    # x = np.array(x)
    # x = x[:, :new_N_train]
    # x = np.reshape(x, [batch_size,n_row_train, window])
    #
    # y = np.array(y)
    # y = y[:, :new_N_train]
    # y = np.reshape(y, [batch_size,n_row_train, window])
    #
    # x_val = np.array(x_val)
    # x_val = x_val[:, :new_N_val]
    # x_val = np.reshape(x_val, [batch_size,n_row_val, window])
    #
    # y_val = np.array(y_val)
    # y_val = y_val[:, :new_N_val]
    # y_val = np.reshape(y_val, [batch_size, n_row_val, window])
    #
    # x_test = np.array(x_test)
    # x_test = x_test[:, :new_N_val]
    # x_test = np.reshape(x_test, [batch_size, n_row_val, window])
    #
    # y_test = np.array(y_test)
    # y_test = y_test[:, :new_N_val]
    # y_test = np.reshape(y_test, [batch_size, n_row_val, window])
    #
    # r = np.array(r)