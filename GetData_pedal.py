import pickle
import random
import os
import tensorflow as tf
import numpy as np
from Preprocess import my_scaler

def get_data(data_dir, w_length, seed=422):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # -----------------------------------------------------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------------------------------------------------
    meta = open(os.path.normpath('/'.join([data_dir, 'metadatasOD300_train.pickle'])), 'rb')
    file_data = open(os.path.normpath('/'.join([data_dir, 'dataOD300_train.pickle'])), 'rb')

    Z = pickle.load(file_data)
    inp = Z['inp']
    tar = Z['tar']

    inp = np.array(inp, dtype=np.float32)
    tar = np.array(tar, dtype=np.float32)
    Z = [inp, tar]
    Z = np.array(Z)

    M = pickle.load(meta)
    tone = M['tone']
    drive = M['drive']
    mode = M['mode']
    fs = M['samplerate']

    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    scaler = my_scaler()
    scaler.fit(Z)

    inp = scaler.transform(inp)
    tar = scaler.transform(tar)

    tone = np.array(tone, dtype=np.float32)
    drive = np.array(drive, dtype=np.float32)
    mode = np.array(mode, dtype=np.float32)

    scaler_metadata = my_scaler()

    scaler_metadata.fit(tone)
    tone = scaler_metadata.transform(tone)
    drive = scaler_metadata.transform(drive)
    mode = scaler_metadata.transform(mode)

    zero_value = (0 - scaler.min_data) / (scaler.max_data - scaler.min_data)
    zero_value_meta = (0 - scaler_metadata.min_data) / (scaler_metadata.max_data - scaler_metadata.min_data)
    zero_value = [zero_value, zero_value_meta]
    scaler = [scaler, scaler_metadata]
    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------

    x, y, x_val, y_val, x_test, y_test = [], [], [], [], [], []

    window = w_length#int(fs * w_length)
    all_inp, all_tar = [], []

    for i in range(inp.shape[0]):
        for t in range(inp.shape[1] // window):
            inp_temp = np.array([inp[i, t * window:t * window + window], np.repeat(tone[i], window),
                                 np.repeat(drive[i], window), np.repeat(mode[i], window)])
            all_inp.append(inp_temp.T)
            tar_temp = np.array(tar[i, t * window:t * window + window])
            all_tar.append(tar_temp.T)

    all_inp = np.array(all_inp)
    all_tar = np.array(all_tar)

    w = 2  # n of column
    h = len(all_inp)  # n of row
    matrix = [[0 for x in range(w)] for y in range(h)]
    for i in range(h):
        matrix[i][0] = all_inp[i]
        matrix[i][1] = all_tar[i]

    N = all_inp.shape[0]
    n_train = N // 100 * 70
    n_val = (N - n_train)

    for n in range(n_train):
        x.append(matrix[n][0])
        y.append(matrix[n][1])

    for n in range(n_val):
        x_val.append(matrix[n_train + n][0])
        y_val.append(matrix[n_train + n][1])

    x = np.array(x)
    y = np.array(y)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    # TEST
    all_inp, all_tar = [], []

    meta = open(os.path.normpath('/'.join([data_dir, 'metadatasOD300_test.pickle'])), 'rb')
    file_data = open(os.path.normpath('/'.join([data_dir, 'dataOD300_test.pickle'])), 'rb')
    Z = pickle.load(file_data)
    inp = Z['inp']
    tar = Z['tar']
    inp = np.array(inp, dtype=np.float32)
    tar = np.array(tar, dtype=np.float32)
    M = pickle.load(meta)
    tone = M['tone']
    drive = M['drive']
    mode = M['mode']

    inp = scaler[0].transform(inp)
    tar = scaler[0].transform(tar)
    tone = np.array(tone, dtype=np.float32)
    drive = np.array(drive, dtype=np.float32)
    mode = np.array(mode, dtype=np.float32)

    tone = scaler[1].transform(tone)
    drive = scaler[1].transform(drive)
    mode = scaler[1].transform(mode)

    for t in range(inp.shape[1] // window):
        inp_temp = np.array(
            [inp[0, t * window:t * window + window], np.repeat(tone[i], window),
                                 np.repeat(drive[i], window), np.repeat(mode[i], window)])
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

    N = all_inp.shape[0]
    for n in range(N):
        x_test.append(matrix[n][0])
        y_test.append(matrix[n][1])

    x_test = np.array(x_test)
    y_test = np.array(y_test)

    return x, y, x_val, y_val, x_test, y_test, scaler, zero_value, fs

if __name__ == '__main__':

    data_dir = '../Files'
    w1 = 1
    w2 = 2
    w4 = 4
    w8 = 8
    w16 = 16

    x, y, x_val, y_val, x_test, y_test, scaler, zero_value, fs = get_data(data_dir=data_dir, w_length=w2, seed=422)

    data = {'x': x, 'y': y, 'x_val': x_val, 'y_val': y_val, 'x_test': x_test, 'y_test': y_test, 'scaler': scaler, 'zero_value': zero_value, 'fs': fs}

    file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_OD300_w1.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()