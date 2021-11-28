import pickle
import random
import os
import tensorflow as tf
import numpy as np
from Code.Preprocess import my_scaler

def get_data(data_dir, seed=422):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    data_dir = '/Users/riccardosimionato/Datasets/VA/VA_results'
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

    scaler = my_scaler()
    scaler.fit(inp)
    scaler.fit(tar)
    inp = scaler.transform(Z)
    tar = scaler.transform(Z)
    zero_value = (0 - scaler.min_data)/(scaler.max_data - scaler.min_data)

    # -----------------------------------------------------------------------------------------------------------------
    # Shuffle indexing matrix and and split into test, train validation
    # -----------------------------------------------------------------------------------------------------------------
    N = len(inp[0])
    n_t = N//100*70
    x, y, x_val, y_val = [],[],[],[]
    for i in inp:
        x.append(i[0:n_t])
        x_val.append(i[n_t+1::])
    for t in tar:
        y.append(t[0:n_t])
        y_val.append(t[n_t + 1::])

    x = np.array(x)
    y = np.array(y)
    x_val = np.array(x_val)
    y_val = np.array(y_val)


    return x, y, x_val, y_val, scaler, zero_value
