import pickle
import random
import os
import tensorflow as tf
import numpy as np
from Code.Preprocess import my_scaler

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

    inp = Z['inp']
    tar = Z['tar']
    # -----------------------------------------------------------------------------------------------------------------
    # Scale data to be within (0, 1)
    # -----------------------------------------------------------------------------------------------------------------

    inp = np.array(inp, dtype=np.float32)
    tar = np.array(tar, dtype=np.float32)

    scaler_inp = my_scaler()
    scaler_tar = my_scaler()
    scaler_inp.fit(inp)
    scaler_tar.fit(tar)
    inp = scaler_inp.transform(inp)
    tar = scaler_tar.transform(tar)
    zero_value_inp = (0 - scaler_inp.min_data)/(scaler_inp.max_data - scaler_inp.min_data)
    zero_value_tar = (0 - scaler_tar.min_data) / (scaler_tar.max_data - scaler_tar.min_data)
    zero_value = [zero_value_inp, zero_value_tar]
    scaler = [scaler_inp, scaler_tar]
    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------
    N = len(inp[0]) #32097856
    #n_t = N//100*70
    n_t = 100
    x, y, x_val, y_val = [], [], [], []
    #batch_size = 1
    inp = inp[0:batch_size, :]
    tar = tar[0:batch_size, :]
    for i in inp:
        x.append(i[0:n_t])
        #x_val.append(i[n_t + 1::])
        x_val.append(i[n_t + 1:200])
    for t in tar:
        y.append(t[0:n_t])
        #y_val.append(t[n_t + 1::])
        y_val.append(t[n_t + 1:200])

    x = np.array(x)
    #x = np.reshape(x, [50*2, -1])
    y = np.array(y)
    #y = np.reshape(x, [50*2, -1])
    x_val = np.array(x_val)
    y_val = np.array(y_val)


    return x, y, x_val, y_val, scaler, zero_value
