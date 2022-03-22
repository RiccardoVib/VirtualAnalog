import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import glob
import pickle

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]

def data_preparation(**kwargs):
    data_dir = '/Users/riccardosimionato/Datasets/VA'
    factor = 2
    save_dir = kwargs.get('save_dir', '/Users/riccardosimionato/Datasets/VA/VA_results')
    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, '*.wav'])))

    L = 31000000

    inp_test, tar_test, ratio_test, threshold_test = [], [], [], []
    fs = 0
    test_rec = False
    for file in file_dirs:
       #733_-40
       #466_-10
        if file == '/Users/riccardosimionato/Datasets/VA/TubeTech_733_-40.wav':
            test_rec = True

        filename = os.path.split(file)[-1]
        metadata = filename.split('_', 2)
        ratio = metadata[1]
        threshold = metadata[-1].replace('.wav', '')
        fs, audio_stereo = wavfile.read(file)
        inp = audio_stereo[:L, 0].astype(np.float32)
        tar = audio_stereo[1:L+1, 1].astype(np.float32)

        ratio = str(ratio)
        if len(ratio) > 2:
            ratio = ratio[:2] + '.' + ratio[2:]

        inp = signal.resample_poly(inp, 1, factor)
        tar = signal.resample_poly(tar, 1, factor)

        ratio = float(ratio)
        threshold = float(threshold)

        if test_rec == True:

            inp_test.append(inp)
            tar_test.append(tar)
            ratio_test.append(ratio)
            threshold_test.append(np.abs(threshold))
            test_rec = False

    metadatas_test = {'ratio': ratio_test, 'threshold': threshold_test, 'samplerate': fs/factor}
    data_test = {'inp': inp_test, 'tar': tar_test}

    file_metadatas_test = open(os.path.normpath('/'.join([save_dir, 'metadatas733_-40.pickle'])), 'wb')
    file_data_test = open(os.path.normpath('/'.join([save_dir, 'data733_-40.pickle'])), 'wb')

    pickle.dump(metadatas_test, file_metadatas_test)
    pickle.dump(data_test, file_data_test)
    # close the file
    file_metadatas_test.close()
    file_data_test.close()


if __name__ == '__main__':

    data_preparation()