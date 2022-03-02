import os
import tensorflow as tf
import numpy as np
import pickle
import os
import glob
from Code.Preprocess import my_scaler_stft
import librosa

from scipy.io import wavfile
from scipy import signal
import Transformer_2
from TrainFunctionality import CustomSchedule, get_batches
from tensorflow.keras.utils import Progbar
import matplotlib.pyplot as plt
from TrainRAMT_2 import train_RAMT
import time

factor = 15
nperseg_i = round(.2 * 44100 / factor)
nfft = 1024

seed=422
data_dir_test = '/Users/riccardosimionato/Datasets/bach_multiinstrumental/TESTMODELS'
save_dir = '/Users/riccardosimionato/Datasets/bach_multiinstrumental/TESTMODELS'

save_model_best = '/Users/riccardosimionato/Datasets/bach_multiinstrumental/TrainedModels/Transformer4h_mse/Run_0/Best'
loaded = tf.saved_model.load(save_model_best)


file_data = open(os.path.normpath('/'.join([data_dir_test, 'data_.pickle'])), 'rb')
file_scaler = open(os.path.normpath('/'.join([data_dir_test, 'scaler.pickle'])), 'rb')

scaler = pickle.load(file_scaler)
data = pickle.load(file_data)
Z = data['data']
Z = np.array(Z)

target_name = '_target.wav'
target_dir = os.path.normpath(os.path.join(save_dir, save_dir, 'WavPredictions', target_name))

if not os.path.exists(os.path.dirname(target_dir)):
    os.makedirs(os.path.dirname(target_dir))

_, target = signal.istft(Z[0].T, nperseg=nperseg_i, nfft=nfft)
target = signal.resample_poly(target, up=15, down=1)
target = target.astype('int16')

wavfile.write(target_dir, 44100, target)

input_name = '_input.wav'
input_dir = os.path.normpath(os.path.join(save_dir, save_dir, 'WavPredictions', input_name))
_, input = signal.istft(Z[1].T, nperseg=nperseg_i, nfft=nfft)
input = signal.resample_poly(input, up=15, down=1)
input = input.astype('int16')

wavfile.write(input_dir, 44100, input)


scaler = scaler['scaler']

#scaler = my_scaler_stft()
scaler.fit(Z)
Z = scaler.transform(Z)


num_layers= 3
d_model= 128
dff= 256
num_heads= 4
drop= 0.1
learning_rate= 0.001
max_length = Z.shape[1]
output_dim = Z.shape[-1]

x = [0,1]
y = [0,1]
transformer = Transformer_2.Transformer(num_layers=num_layers,
                                      d_model=d_model,
                                      num_heads=num_heads,
                                      dff=dff,  # Hidden layer size of feedforward networks
                                      input_vocab_size=None,  # Not relevant for ours as we don't use embedding
                                      target_vocab_size=None,
                                      pe_input=max_length,  # Max length for positional encoding input
                                      pe_target=max_length,
                                      output_dim=output_dim,
                                      rate=drop)  # Dropout rate



# Need to make a single prediction for the model as it needs to compile:
transformer([tf.constant(Z[x[:1]], dtype='float32'), tf.constant(Z[y[:1], :-1, :], dtype='float32')], training=False)

for i in range(len(transformer.variables)):
    if transformer.variables[i].name != loaded.all_variables[i].name:
        assert ValueError('Cannot load model, due to incompatible loaded and model...')
        transformer.variables[i].assign(loaded.all_variables[i].value())

if 'n_params' not in locals():
    n_params = np.sum([np.prod(v.get_shape()) for v in transformer.variables])


predictions = transformer([tf.constant(Z[x[1:]], dtype='float32'), tf.constant(Z[y[:1], :-1, :], dtype='float32')], training=False)

predictions = predictions[0].numpy()



predictions = scaler.inverse_transform(predictions)
pred_name = '_pred.wav'

pred_dir = os.path.normpath(os.path.join(save_dir, save_dir, 'WavPredictions', pred_name))
y_dir = os.path.normpath(os.path.join(save_dir, save_dir, 'WavPredictions', '_y.wav'))
if not os.path.exists(os.path.dirname(pred_dir)):
    os.makedirs(os.path.dirname(pred_dir))

y = librosa.griffinlim(predictions[0], n_iter=132)

fig, ax = plt.subplots(nrows=1, sharex=True, sharey=True)
ax.plot(y, 'b--', label='Target')
plt.show()


_, pred_i = signal.istft(predictions[0].T, nperseg=nperseg_i, nfft=nfft)
pred_i = signal.resample_poly(pred_i, up=15, down=1)
pred_i = pred_i.astype('int16')


wavfile.write(pred_dir, 44100, pred_i)

fig, axs = plt.subplots()
#pcm = axs.pcolormesh(np.arange(predictions[0].shape[0]), np.arange(predictions[0].shape[-1]), predictions[0].T, vmin=0,
#                    vmax=np.max(predictions[0].T), shading='gouraud')

pcm = axs.pcolormesh(np.arange(Z[1].shape[0]), np.arange(Z[1].shape[-1]), Z[1].T, vmin=0,
                     vmax=np.max(Z[1].T), shading='gouraud')
fig.colorbar(pcm)
axs.set_title('Test with Arbitrary Sample')
axs.set_ylabel('Frequency [Hz]')
axs.set_xlabel('Time [sec]')
plt.show()