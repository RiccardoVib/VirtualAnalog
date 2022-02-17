import numpy as np
from gtda.time_series import TakensEmbedding
import gtda.time_series
import pickle
import os

data_dir = '../Files'
file_data = open(os.path.normpath('/'.join([data_dir, 'data.pickle'])), 'rb')
Z = pickle.load(file_data)
inp = Z['inp']
tar = Z['tar']

inp = np.array(inp, dtype=np.float32)
tar = np.array(tar, dtype=np.float32)

from teaspoon.parameter_selection.FNN_n import FNN_n
import numpy as np



tau=15 #embedding delay

perc_FNN, n = FNN_n(tar[12,:], tau, plotting = True)
print('FNN embedding Dimension: ',n)

#gtda.time_series.takens_embedding_optimal_parameters(tar[1,:], max_time_delay=32, max_dimension=32, stride=1, n_jobs=None, validate=True)