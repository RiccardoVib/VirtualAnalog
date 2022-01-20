import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy import fft
import os
import glob
import matplotlib.pyplot as plt

#data_dir_dense = 'C:/Users/riccarsi/Documents/GitHub/Results/DenseFeed_Testing/WavPredictions'
#data_dir_dense = 'C:/Users/riccarsi/Documents/GitHub/Results/DenseFeed_Testing/WavPredictions'
#data_dir_LSTM = 'C:/Users/riccarsi/Documents/GitHub/Results/LSTM2_Testing/WavPredictions'
#data_dir_LSTM = 'C:/Users/riccarsi/Documents/GitHub/Results/LSTM2_Testing/WavPredictions'

file_tar_dense = glob.glob(os.path.normpath('/'.join([data_dir_dense, '_tar.wav'])))
file_tar_LSTM = glob.glob(os.path.normpath('/'.join([data_dir_LSTM, '_tar.wav'])))
file_pred_dense = glob.glob(os.path.normpath('/'.join([data_dir_dense, '_pred.wav'])))
file_pred_LSTM = glob.glob(os.path.normpath('/'.join([data_dir_LSTM, '_pred.wav'])))

for file in file_tar_dense:
    fs, audio_tar_dense = wavfile.read(file)
for file in file_pred_dense:
    _, audio_pred_dense = wavfile.read(file)
for file in file_tar_LSTM:
    fs, audio_tar_LSTM = wavfile.read(file)
for file in file_pred_LSTM:
    _, audio_pred_LSTM = wavfile.read(file)

#Dense
audio_tar_dense = audio_tar_dense.astype(np.float32)
audio_pred_dense = audio_pred_dense.astype(np.float32)
audio_tar_dense = audio_tar_dense[:len(audio_pred_dense)]

time = np.linspace(0, len(audio_tar_dense) / fs, num=len(audio_tar_dense))
N = len(audio_tar_dense)
fs = 1600
fft_tar = fft.fftshift(fft.fft(audio_tar_dense))[N//2:]
fft_pred = fft.fftshift(fft.fft(audio_pred_dense))[N//2:]
freqs = fft.fftshift(fft.fftfreq(N)*fs)
freqs = freqs[N//2:]

fig, ax = plt.subplots()
plt.title("Target vs Predictionn - Time Domain")
ax.plot(time, audio_tar_dense, label='Target')
ax.plot(time, audio_pred_dense, label='Prediction')
ax.set_xlabel('Time')
ax.set_ylabel('Amplitude')
ax.legend()
plt.show()

fig, ax = plt.subplots()
plt.title("Target vs Predictionn - Frequency Domain")
ax.plot(freqs, fft_tar, label='Target')
ax.plot(freqs, fft_pred, label='Prediction')
ax.set_xlabel('Frequency')
ax.set_ylabel('Amplitude')
ax.legend()
plt.show()