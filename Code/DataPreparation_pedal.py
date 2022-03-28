import numpy as np
from scipy.io import wavfile
from scipy import signal
import os
import glob
import pickle
import audio_format
def get_keys_from_value(d, val):
    return [k for k, v in d.items() if v == val]
#
def data_preparation(**kwargs):
    data_dir = '/Users/riccardosimionato/Datasets/OD300/Wav'
    factor = 2
    #data_dir = kwargs.get('data_dir', '/Users/riccardosimionato/Datasets/OD300/Wav')
    save_dir = kwargs.get('save_dir', '/Users/riccardosimionato/Datasets/OD300/OD300_results')
    file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, '*.wav'])))


    #L = 32000000#se 48kHz
    #L = 31000000
    L = 20000000

    inp_collector, tar_collector, tone_collector, drive_collector, mode_collector = [], [], [], [], []
    inp_test, tar_test, tone_test, drive_test, mode_test = [], [], [], [], []
    fs = 0#fs= 96,000 Hz

    inp_collector_never_seen, tar_collector_never_seen, tone_collector_never_seen, drive_collector_never_seen, mode_collector_never_seen  = [], [], [], [], []
    test_rec = False

    #tone 0-2
    #drive 0-2
    #mode ds,md,od 0-2

    for file in file_dirs:

        if file == '/Users/riccardosimionato/Datasets/OD300/Wav/OD300_Tone_1_Drive_1_Mode_mid.wav':
            test_rec = True

        filename = os.path.split(file)[-1]
        metadata = filename.split('_', 6)
        tone = metadata[2]
        drive = metadata[4]
        mode = metadata[-1].replace('.wav', '')

        if mode == 'ds':
            mode = 0
        elif mode == 'mid':
            mode = 1
        elif mode == 'od':
            mode = 2
        else:
            raise ValueError('Problems!')

        fs, audio_stereo = wavfile.read(file)

        inp = audio_format.pcm2float(audio_stereo[:L, 0])
        tar = audio_format.pcm2float(audio_stereo[1:L+1, 1])

        #inp = audio_stereo[:L, 0].astype(np.float32)
        #tar = audio_stereo[1:L+1, 1].astype(np.float32)

        inp_never_seen = audio_format.pcm2float(audio_stereo[L:, 0])
        tar_never_seen = audio_format.pcm2float(audio_stereo[L+1:, 1])

        inp = signal.resample_poly(inp, 1, factor)
        tar = signal.resample_poly(tar, 1, factor)

        inp_never_seen = signal.resample_poly(inp_never_seen, 1, factor)
        tar_never_seen = signal.resample_poly(tar_never_seen, 1, factor)

        tone = float(tone)/2
        drive = float(drive)/2
        mode = float(mode)/2
        #tar = np.pad(tar, (1, 0), mode='constant', constant_values=0)

        inp_collector_never_seen.append(inp_never_seen)
        tar_collector_never_seen.append(tar_never_seen)
        tone_collector_never_seen.append(tone)
        drive_collector_never_seen.append(drive)
        mode_collector_never_seen.append(mode)

        if test_rec == False:
            inp_collector.append(inp)
            tar_collector.append(tar)
            tone_collector.append(tone)
            drive_collector.append(drive)
            mode_collector.append(mode)
        else:
            inp_test.append(inp)
            tar_test.append(tar)
            tone_test.append(tone)
            drive_test.append(drive)
            mode_test.append(mode)
            test_rec = False

    # train files
    metadatas = {'tone': tone_collector, 'drive': drive_collector, 'mode': mode_collector, 'samplerate': fs/factor}
    data = {'inp': inp_collector, 'tar': tar_collector}
    # test files
    metadatas_test = {'tone': tone_test, 'drive': drive_test, 'mode': mode_test, 'samplerate': fs/factor}
    data_test = {'inp': inp_test, 'tar': tar_test}
    # never seen files
    metadatas_never_seen = {'tone': tone_collector_never_seen, 'drive': drive_collector_never_seen, 'mode': mode_collector_never_seen, 'samplerate': fs/factor}
    data_never_seen = {'inp': inp_collector_never_seen, 'tar': tar_collector_never_seen}

    # open a file, where you ant to store the data
    # train files
    file_metadatas = open(os.path.normpath('/'.join([save_dir,'metadatasOD300_train.pickle'])), 'wb')
    file_data = open(os.path.normpath('/'.join([save_dir,'dataOD300_train.pickle'])), 'wb')
    # test files
    file_metadatas_test = open(os.path.normpath('/'.join([save_dir,'metadatasOD300_test.pickle'])), 'wb')
    file_data_test = open(os.path.normpath('/'.join([save_dir,'dataOD300_test.pickle'])), 'wb')
    # never seen files
    file_metadatas_never_seen = open(os.path.normpath('/'.join([save_dir,'metadatasOD300_never_seen.pickle'])), 'wb')
    file_data_never_seen = open(os.path.normpath('/'.join([save_dir,'dataOD300_never_seen.pickle'])), 'wb')

    # dump information to that file
    #train files
    pickle.dump(metadatas, file_metadatas)
    pickle.dump(data, file_data)
    #test files
    pickle.dump(metadatas_test, file_metadatas_test)
    pickle.dump(data_test, file_data_test)
    #never seen files
    pickle.dump(metadatas_never_seen, file_metadatas_never_seen)
    pickle.dump(data_never_seen, file_data_never_seen)

    # close the file
    file_metadatas.close()
    file_data.close()
    file_metadatas_test.close()
    file_data_test.close()
    file_metadatas_never_seen.close()
    file_data_never_seen.close()

if __name__ == '__main__':

    data_preparation()