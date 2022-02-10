import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import glob
import pickle


def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


def data_preparation(**kwargs):
    data_dir = '/Users/riccardosimionato/Datasets/Piano'
    #data_dir = 'C:/Users/riccarsi/Documents/GitHub/VA'
    #save_dir = 'C:/Users/riccarsi/Documents/GitHub/VA_pickle'
    save_dir = '/Users/riccardosimionato/Datasets/Piano'
    factor = 6#3

    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, '*.wav'])))
    inp_collector, tar_collector = [], []

    for file in file_dirs:
        filename = os.path.split(file)[-1]
        fs, audio_stereo = wavfile.read(file) #fs= 96,000 Hz
        audio_stereo = audio_stereo.astype(np.float32)
        audio_stereo = signal.resample_poly(audio_stereo, 1, factor)

        if filename == 'Piano.wav':
            tar_collector.append(audio_stereo)
        if filename == 'Sine.wav':
            inp_collector.append(audio_stereo)

    data = {'inp': inp_collector, 'tar': tar_collector}
    file_data = open(os.path.normpath('/'.join([save_dir,'piano_data.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()

if __name__ == '__main__':

    data_preparation()