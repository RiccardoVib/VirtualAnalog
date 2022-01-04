import numpy as np
from scipy.io import wavfile
from scipy import signal
from scipy import fft
import os
import glob
import matplotlib.pyplot as plt

data_dir = 'C:/Users/riccarsi/Documents/GitHub/Results/DenseFeed_Testing/WavPredictions'
data_dir = 'C:/Users/riccarsi/Documents/GitHub/Results/LSTM2_Testing/WavPredictions'

file_tar = glob.glob(os.path.normpath('/'.join([data_dir, '_tar.wav'])))
file_pred = glob.glob(os.path.normpath('/'.join([data_dir, '_pred.wav'])))

for file in file_tar:
    fs, audio_tar = wavfile.read(file)
for file in file_pred:
    _, audio_pred = wavfile.read(file)

audio_tar = audio_tar.astype(np.float32)
audio_pred = audio_pred.astype(np.float32)
audio_tar = audio_tar[:len(audio_pred)]

time = np.linspace(0, len(audio_tar) / fs, num=len(audio_tar))
N = len(audio_tar)
fs = 1600
fft_tar = fft.fftshift(fft.fft(audio_tar))[N//2:]
fft_pred = fft.fftshift(fft.fft(audio_pred))[N//2:]
freqs = fft.fftshift(fft.fftfreq(N)*fs)
freqs = freqs[N//2:]

fig, ax = plt.subplots()
plt.title("Target vs Predictionn - Time Domain")
ax.plot(time, audio_tar, label='Target')
ax.plot(time, audio_pred, label='Prediction')
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