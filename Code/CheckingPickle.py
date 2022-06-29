import pickle
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from audio_format import pcm2float

data_dir = '../Files'

file_data = open(os.path.normpath('/'.join([data_dir, 'data_prepared_w16.pickle'])), 'rb')

Z = pickle.load(file_data)

x = Z['x']



file_data = open(os.path.normpath('/'.join([data_dir, 'Transf_pred_2.wav'])), 'rb')
fs, audio_stereo = wavfile.read(file_data)
pred = audio_stereo.T
file_data = open(os.path.normpath('/'.join([data_dir, 'Transf_tar.wav'])), 'rb')
fs, audio_stereo = wavfile.read(file_data)
tar = audio_stereo

time = np.linspace(0, len(pred) / fs, num=len(pred))
plt.figure()
plt.title("Input")
plt.plot(time, pcm2float(pred), time, pcm2float(tar))
plt.show()

file_data = open(os.path.normpath('/'.join([data_dir, 'Seq2Seq_pred.wav'])), 'rb')
fs, audio_stereo = wavfile.read(file_data)
pred = audio_stereo.T
file_data = open(os.path.normpath('/'.join([data_dir, 'Seq2Seq_tar.wav'])), 'rb')
fs, audio_stereo = wavfile.read(file_data)
tar = audio_stereo

time = np.linspace(0, len(pred) / fs, num=len(pred))
plt.figure()
plt.title("Input")
plt.plot(time, pcm2float(pred), time, pcm2float(tar))
plt.show()
