import librosa
import os
import glob
import numpy as np
from scipy.io import wavfile
from scipy import signal

data_dir = '/Users/riccardosimionato/OneDrive - Universitetet i Oslo/RAMT/TransLSTM_RawAudio-main/SoundSamples/Sample_(a)'
file_dirs = glob.glob(os.path.normpath('/'.join([data_dir, 'Prediction_LSTM.wav'])))

factor = 15
ntime = round(.2*44100/factor)
nfft = 1024
nfreq = 1024

for file in file_dirs:
    samplerate, audio_stereo = wavfile.read(file)

    #inp_i_d = signal.resample_poly(audio_stereo, 1, factor)
    #f, t, Zxx = signal.stft(inp_i_d, fs=samplerate / factor, nperseg=ntime, nfft=nfft)
    f, t, Zxx = signal.stft(audio_stereo, fs=samplerate / factor, nperseg=ntime, nfft=nfft)
    Zxx_r = np.abs(np.real(Zxx)).T
    Zxx_i = np.imag(Zxx).T


#y = librosa.griffinlim(Zxx_r, n_iter=32, win_length=ntime, n_fft=nfft)
y = librosa.griffinlim(Zxx_r.T, n_iter=32)


import matplotlib.pyplot as plt

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True)
ax[0].plot(audio_stereo, 'b--', label='Target')
ax[1].plot(y, 'r:', label='Prediction')
plt.show()