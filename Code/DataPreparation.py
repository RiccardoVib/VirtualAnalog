import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import glob
import pickle
#import matplotlib.pyplot as plt
#import wave

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def data_preparation(**kwargs):
    #data_dir = '/Users/riccardosimionato/Datasets/VA'
    data_dir = 'C:/Users/riccarsi/Documents/GitHub/VA'
    save_dir = 'C:/Users/riccarsi/Documents/GitHub/VA_pickle'
    factor = 6#3
    #data_dir = kwargs.get('data_dir', '/Users/riccardosimionato/Datasets/VA')
    #save_dir = kwargs.get('save_dir', '/Users/riccardosimionato/Datasets/VA/VA_results')
    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, '*.wav'])))

    L = 5349643-100#10699286-100 #32097856#MAX=34435680
    inp_collector, tar_collector, ratio_collector, threshold_collector = [], [], [], []
    ratio, threshold = '', ''
    fs = 0
    for file in file_dirs:

        filename = os.path.split(file)[-1]
        metadata = filename.split('_', 2)
        ratio = metadata[1]
        threshold = metadata[-1].replace('.wav', '')
        fs, audio_stereo = wavfile.read(file) #fs= 96,000 Hz
        inp = audio_stereo[:, 0].astype(np.float32)
        tar = audio_stereo[1:, 1].astype(np.float32)
        ratio = str(ratio)
        if len(ratio) > 2:
            ratio = ratio[:2] + '.' + ratio[2:]

        #target is delayed by one sample due the system processing so
        #need to be moved
        tar = tar[1:len(tar)]

        inp = signal.resample_poly(inp, 1, factor)
        tar = signal.resample_poly(tar, 1, factor)
        ratio = float(ratio)
        threshold = float(threshold)
        #tar = np.pad(tar, (1, 0), mode='constant', constant_values=0)

        if len(tar) > L:
            inp = inp[0:L]
            tar = tar[0:L]

        #if len(inp) < L:
        #    L = len(inp)

        inp_collector.append(inp)
        tar_collector.append(tar)
        ratio_collector.append(ratio)
        threshold_collector.append(np.abs(threshold))
#        time = np.linspace(0, len(inp) / fs, num=len(inp))
#        plt.figure()
#        plt.title("Input")
#        plt.plot(time, inp)
#        plt.show()

#        plt.figure()
#        plt.title("Target")
#        plt.plot(time, tar)
#        plt.show()

    metadatas = {'ratio': ratio_collector, 'threshold': threshold_collector, 'samplerate': fs/factor}
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