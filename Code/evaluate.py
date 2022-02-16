import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy import fft
import os
import glob
import matplotlib.pyplot as plt
from TrainFunctionality import error_to_signal_ratio

def plot_result(data_dir, save):
    file_tar = glob.glob(os.path.normpath('/'.join([data_dir, '*_tar.wav'])))
    file_pred = glob.glob(os.path.normpath('/'.join([data_dir, '*_pred.wav'])))

    for file in file_tar:
        fs, audio_tar = wavfile.read(file)
    for file in file_pred:
        _, audio_pred = wavfile.read(file)

    audio_tar = audio_tar.astype(np.float32)
    audio_pred = audio_pred[0:1600].astype(np.float32)

    for index in range(len(audio_tar)//16):

        audio_tar = np.delete(audio_tar, index)
        index *= 16

    audio_tar = audio_tar[:len(audio_pred)]

    print(error_to_signal_ratio(audio_tar, audio_pred))

    #audio_tar_dense = audio_tar_dense[1600:1600*2]
    #audio_pred_dense = audio_pred_dense[1600:1600*2]

    time = np.linspace(0, len(audio_tar) / fs, num=len(audio_tar))
    N = len(audio_tar)
    fs = 1600
    fft_tar = fft.fftshift(fft.fft(audio_tar))[N//2:]
    fft_pred = fft.fftshift(fft.fft(audio_pred))[N//2:]
    freqs = fft.fftshift(fft.fftfreq(N)*fs)
    freqs = freqs[N//2:]

    fig, ax = plt.subplots()
    plt.title("Target vs Predictionn - Time Domain")
    ax.plot(time, audio_tar, 'b--', label='Target')
    ax.plot(time, audio_pred, 'r:', label='Prediction')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.show()

    fname = os.path.normpath('/'.join([data_dir, 'time.png']))

    if save:
        fig.savefig(fname)

    fig, ax = plt.subplots()
    plt.title("Target vs Predictionn - Frequency Domain")
    ax.plot(freqs, fft_tar, 'b--', label='Target')
    ax.plot(freqs, fft_pred, 'r:', label='Prediction')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Amplitude')
    ax.legend()
    plt.show()

    fname = os.path.normpath('/'.join([data_dir, 'freq.png']))

    if save:
        fig.savefig(fname)

    #plt.close(fig)

if __name__ == '__main__':
    data_dir_ed = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_enc_dec_no_sig_Testing/WavPredictions'
    data_dir_eds = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_Testing_enc_dec_sig/WavPredictions'
    data_dir_LSTM = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_Testing_normal_no_sig/WavPredictions'

    plot_result(data_dir=data_dir_ed, save=True)
    plot_result(data_dir=data_dir_eds, save=True)
    plot_result(data_dir=data_dir_LSTM, save=True)