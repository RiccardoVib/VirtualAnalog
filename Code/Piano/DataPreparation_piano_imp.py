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
    inp_collector_imp, tar_collector_imp = [], []
    inp_collector_sine, tar_collector_sine = [], []

    for file in file_dirs:
        filename = os.path.split(file)[-1]
        fs, audio_stereo = wavfile.read(file) #fs= 96,000 Hz
        audio_stereo = audio_stereo.astype(np.float32)
        audio_stereo = signal.resample_poly(audio_stereo, 1, factor)

        if filename == 'Piano.wav':
            tar_collector.append(audio_stereo)
        if filename == 'Sine.wav':
            inp_collector.append(audio_stereo)

        if filename == 'Piano_per_imp.wav':
            tar_collector_sine.append(audio_stereo)
        if filename == 'Sine_per_imp.wav':
            inp_collector_sine.append(audio_stereo)

        if filename == 'Piano_per_imp.wav':
            tar_collector_imp.append(audio_stereo)
        if filename == 'square.wav':
            inp_collector_imp.append(audio_stereo)


    data = {'inp': inp_collector, 'tar': tar_collector}
    data_imp = {'inp': inp_collector_imp, 'tar': tar_collector_imp}
    data_sine = {'inp': inp_collector_sine, 'tar': tar_collector_sine}
    file_data = open(os.path.normpath('/'.join([save_dir,'piano_data.pickle'])), 'wb')
    file_data_imp = open(os.path.normpath('/'.join([save_dir,'piano_data_imp.pickle'])), 'wb')
    file_data_sine = open(os.path.normpath('/'.join([save_dir,'piano_data_sine.pickle'])), 'wb')
    pickle.dump(data, file_data)
    pickle.dump(data_imp, file_data_imp)
    pickle.dump(data_sine, file_data_sine)
    file_data.close()
    file_data_imp.close()
    file_data_sine.close()

if __name__ == '__main__':

    data_preparation()