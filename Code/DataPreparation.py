import numpy as np
from scipy.io import wavfile
import os
import glob
import pickle
import matplotlib.pyplot as plt
import wave

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def data_preparation(**kwargs):
    data_dir = '/Users/riccardosimionato/Datasets/VA'

    data_dir = kwargs.get('data_dir', '/Users/riccardosimionato/Datasets/VA')
    save_dir = kwargs.get('save_dir', '/Users/riccardosimionato/Datasets/VA/VA_results')
    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, '*.wav'])))

    L = 32097856#MAX=34435680
    inp_collector, tar_collector = [],[]
    for file in file_dirs:

        filename = os.path.split(file)[-1]
        metadata = filename.split('_', 2)
        ratio = metadata[1]
        threshold = metadata[-1].replace('.wav', '')
        fs, audio_stereo = wavfile.read(file)
        inp = audio_stereo[:, 0].astype(np.float32)
        tar = audio_stereo[:, 1].astype(np.float32)

        if len(inp) > L:
            inp = inp[0:L]
            tar = tar[0:L]

        inp_collector.append(inp)
        tar_collector.append(tar)
#        time = np.linspace(0, len(inp) / fs, num=len(inp))
#        plt.figure()
#        plt.title("Input")
#        plt.plot(time, inp)
#        plt.show()

#        plt.figure()
#        plt.title("Target")
#        plt.plot(time, tar)
#        plt.show()

    metadatas = {'ratio': ratio, 'threshold': threshold, 'samplerate': fs}
    data = {'inp': inp_collector, 'tar': tar_collector}

    # open a file, where you ant to store the data
    file_metadatas = open(os.path.normpath('/'.join([save_dir,'metadatas.pickle'])), 'wb')
    file_data = open(os.path.normpath('/'.join([save_dir,'data.pickle'])), 'wb')

    # dump information to that file
    pickle.dump(metadatas, file_metadatas)
    pickle.dump(data, file_data)

    # close the file
    file_metadatas.close()
    file_data.close()

if __name__ == '__main__':

    data_preparation()