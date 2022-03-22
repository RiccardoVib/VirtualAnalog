import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import glob
import pickle


# import matplotlib.pyplot as plt
# import wave

def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]


def data_preparation(**kwargs):
    data_dir = '/Users/riccardosimionato/Datasets/VA'
    # data_dir = 'C:/Users/riccarsi/Documents/GitHub/VA'
    # save_dir = 'C:/Users/riccarsi/Documents/GitHub/VA_pickle'
    factor = 2  # 6#3
    # data_dir = kwargs.get('data_dir', '/Users/riccardosimionato/Datasets/VA')
    save_dir = kwargs.get('save_dir', '/Users/riccardosimionato/Datasets/VA/VA_results')
    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, '*.wav'])))

    # L = 32000000#se 48kHz
    L = 31000000

    # L = 32097856##10699286-100 #32097856#MAX=34435680
    # L = 5349643-100 #se 16kHz
    inp_collector, tar_collector, ratio_collector, threshold_collector = [], [], [], []
    fs = 0

    inp_collector_never_seen, tar_collector_never_seen, ratio_collector_never_seen, threshold_collector_never_seen = [], [], [], []
    metadatas = []
    data =[]
    test_rec = False
    for file in file_dirs:

        if file == '/Users/riccardosimionato/Datasets/VA/TubeTech_333_-30.wav':
            test_rec = True

        filename = os.path.split(file)[-1]
        metadata = filename.split('_', 2)
        ratio = metadata[1]
        threshold = metadata[-1].replace('.wav', '')
        fs, audio_stereo = wavfile.read(file)  # fs= 96,000 Hz
        inp = audio_stereo[:L, 0].astype(np.float32)
        tar = audio_stereo[1:L + 1, 1].astype(np.float32)

        inp_never_seen = audio_stereo[L:, 0].astype(np.float32)
        tar_never_seen = audio_stereo[L + 1:, 1].astype(np.float32)

        ratio = str(ratio)
        if len(ratio) > 2:
            ratio = ratio[:2] + '.' + ratio[2:]

        # target is delayed by one sample due the system processing so
        # need to be moved
        # tar = tar[1:len(tar)]

        inp = signal.resample_poly(inp, 1, factor)
        tar = signal.resample_poly(tar, 1, factor)

        #inp_never_seen = signal.resample_poly(inp_never_seen, 1, factor)
        #tar_never_seen = signal.resample_poly(tar_never_seen, 1, factor)

        ratio = float(ratio)
        threshold = float(threshold)

        # inp_collector_never_seen.append(inp_never_seen)
        # tar_collector_never_seen.append(tar_never_seen)
        # ratio_collector_never_seen.append(ratio)
        # threshold_collector_never_seen.append(np.abs(threshold))

        # find
        sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
        sec = [32, 135, 238, 240.9, 308.7]
        sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]
        start = np.zeros(len(sig_name), dtype=int)
        stop = np.zeros(len(sig_name), dtype=int)
        for l in range(len(sig_name)):
            start[l] = int(sec[l] * fs // factor)
            stop[l] = int(sec_end[l] * start[l])

        sweep_inp = inp[start[0]:stop[0]]
        guitar_inp = inp[start[1]:stop[1]]
        drumKick_inp = inp[start[2]:stop[2]]
        drumHH_inp = inp[start[3]:stop[3]]
        bass_inp = inp[start[4]:stop[4]]
        sweep_tar = tar[start[0]:stop[0]]
        guitar_tar = tar[start[1]:stop[1]]
        drumKick_tar = tar[start[2]:stop[2]]
        drumHH_tar = tar[start[3]:stop[3]]
        bass_tar = tar[start[4]:stop[4]]

        if test_rec == False:
            _metadatas = {'ratio': ratio, 'threshold': threshold, 'samplerate': fs / factor}
            metadatas.append(_metadatas)
            _data = {'sweep_inp': sweep_inp, 'sweep_tar': sweep_tar,
                     'guitar_inp': guitar_inp, 'guitar_tar': guitar_tar,
                     'drumKick_inp': drumKick_inp, 'drumKick_tar': drumKick_tar,
                     'drumHH_inp': drumHH_inp, 'drumHH_tar': drumHH_tar,
                     'bass_inp': bass_inp, 'bass_tar': bass_tar}
            data.append(_data)
        else:
            _metadatas = {'ratio_test': ratio, 'threshold_test': threshold, 'samplerate': fs / factor}
            metadatas.append(_metadatas)
            _data = {'sweep_inp_test': sweep_inp, 'sweep_tar_test': sweep_tar,
                    'guitar_inp_test': guitar_inp, 'guitar_tar_test': guitar_tar,
                    'drumKick_inp_test': drumKick_inp, 'drumKick_tar_test': drumKick_tar,
                    'drumHH_inp_test': drumHH_inp, 'drumHH_tar_test': drumHH_tar,
                    'bass_inp_test': bass_inp, 'bass_tar_test': bass_tar}
            data.append(_data)
            test_rec = False
    # files
    file_metadatas = open(os.path.normpath('/'.join([save_dir, 'meta_samples_collection.pickle'])), 'wb')
    file_data = open(os.path.normpath('/'.join([save_dir, 'test_samples_collection.pickle'])), 'wb')

    # dump information to that file
    pickle.dump(metadatas, file_metadatas)
    pickle.dump(data, file_data)

    # close the file
    file_metadatas.close()
    file_data.close()

if __name__ == '__main__':
    data_preparation()