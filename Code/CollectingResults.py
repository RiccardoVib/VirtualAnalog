from ProperEvaluationAllModels import load_audio, prediction_accuracy, measure_performance, measure_time, load_model_dense, load_model_lstm
from ProperEvaluationAllModels import load_model_lstm_enc_dec, load_model_lstm_enc_dec_v2, inferenceLSTM_enc_dec, inferenceLSTM_enc_dec_v2
from ProperEvaluationAllModels import plot_time, plot_fft, create_ref
import os
import pickle
import glob
import audio_format
from scipy.io import wavfile

data_ = '../Files'
file_data = open(os.path.normpath('/'.join([data_, 'data_prepared_w1_limited.pickle'])), 'rb')
data = pickle.load(file_data)
x_test = data['x_test']
y_test = data['y_test']
fs = data['fs']
scaler = data['scaler']

#create_ref()
# Dense-----------------------------------------------------------------------------------

# data_dir = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/Dense_trials/DenseFeed_Testing_64_in1'
# name = 'Dense'
# T=x_test.shape[1]
# audio_tar, audio_pred, fs = load_audio(data_dir)
# prediction_accuracy(audio_tar, audio_pred, fs, data_dir, name)
# results = measure_performance(audio_tar, audio_pred, name)
# model = load_model_dense(T, [64, 64], 0., model_save_dir=data_dir)
# time_s = measure_time(model, x_test, y_test, False, False, data_dir, fs, scaler, T)
# with open(os.path.normpath('/'.join([data_dir, 'performance_results.txt'])), 'w') as f:
#    for key, value in results.items():
#        print('\n', key, '  : ', value, file=f)

# LSTM-----------------------------------------------------------------------------------

# data_dir = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_trials/LSTM_Testing_64h'
# name = 'LSTM'
# T=x_test.shape[1]
# audio_tar, audio_pred, fs = load_audio(data_dir)
# prediction_accuracy(audio_tar, audio_pred, fs, data_dir, name)
# results = measure_performance(audio_tar, audio_pred, name)
# model = load_model_lstm(T, [64,64], 0., model_save_dir=data_dir)
# time_s = measure_time(model, x_test, y_test, False, False, data_dir, fs, scaler, T)
#with open(os.path.normpath('/'.join([data_dir, 'performance_results.txt'])), 'w') as f:
#    for key, value in results.items():
#        print('\n', key, '  : ', value, file=f)
# --------------------------------------------------------------------------------------
# change of dataset
# --------------------------------------------------------------------------------------

file_data = open(os.path.normpath('/'.join([data_, 'data_prepared_w2.pickle'])), 'rb')
data = pickle.load(file_data)
x_test = data['x_test']
y_test = data['y_test']
fs = data['fs']
scaler = data['scaler']

# LSTM_enc_dec----------------------------------------------------------------------------
#
# data_dir = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_enc_dec_trials'
# name = 'LSTM_enc_dec'
# T = x_test.shape[1]

# encoder_model, decoder_model = load_model_lstm_enc_dec(T, enc_units, dec_units, drop, model_save_dir=data_dir)
# model = [encoder_model, decoder_model]
# time_s = measure_time(model, x_test, y_test, False, False, data_dir, fs, scaler, T)


# name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
# sec = [32, 135, 238, 240.9, 308.7]
# sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]
# for l in range(len(name)):
#     start = int(sec[l] * fs)
#     end = int(sec_end[l] * start)
    # inferenceLSTM_enc_dec(data_dir, model, fs, x_test, y_test, T, scaler, start, end, name[l], True)


# prediction_accuracy(audio_tar, audio_pred, fs, data_dir, name)
# results = measure_performance(audio_tar, audio_pred, name)

# LSTM_enc_dec_v2-------------------------------------------------------------------------
data_dir_ref='/Users/riccardosimionato/PycharmProjects/All_Results'
data_dir = '/Users/riccardosimionato/PycharmProjects/All_Results/Giusti/LSTM_enc_dec_v2_trials/LSTM_enc_dec_v2_2'
name = 'LSTM_enc_dec_v2'
T = x_test.shape[1]
enc_units = [8]
dec_units = [8]

#model = load_model_lstm_enc_dec_v2(T=T, encoder_units=enc_units, decoder_units=dec_units, drop=0., model_save_dir=data_dir)
#time_s = measure_time(model=model, x_test=x_test, y_test=x_test, enc_dec=True, v2=True, data_dir=data_dir, fs=fs, scaler=scaler, T=T)

sig_name = ['_sweep_', '_guitar_', '_drumKick_', '_drumHH_', '_bass_']
sec = [32, 135, 238, 240.9, 308.7]
sec_end = [1.5, 1.019, 1.0025, 1.0018, 1.007]
for l in range(len(sig_name)):
    start = int(sec[l] * fs)
    end = int(sec_end[l] * start)
    #inferenceLSTM_enc_dec_v2(data_dir=data_dir, x_test=x_test, y_test=y_test, model=model, fs=fs, scaler=scaler, start=start, end=end, T=T, name=name[l], generate=True)

all_results = []
for l in range(len(sig_name)):
    file_tar = glob.glob(os.path.normpath('/'.join([data_dir_ref, sig_name[l] + '_tar.wav'])))
    file_pred = glob.glob(os.path.normpath('/'.join([data_dir, sig_name[l] + '_pred.wav'])))
    for file in file_tar:
        fs, audio_tar = wavfile.read(file)
    for file in file_pred:
        _, audio_pred = wavfile.read(file)

    audio_tar = audio_format.pcm2float(audio_tar)
    audio_tar = audio_tar[:-2]
    audio_pred = audio_format.pcm2float(audio_pred)
    results = measure_performance(audio_tar, audio_pred, name)
    all_results.append(results)
    plot_time(audio_tar, audio_pred, fs, data_dir, 'LSTM_enc_dec_v2' + sig_name[l])
    plot_fft(audio_tar, audio_pred, fs, data_dir, 'LSTM_enc_dec_v2' + sig_name[l])

with open(os.path.normpath('/'.join([data_dir, 'performance_results.txt'])), 'w') as f:
    i=0
    for res in all_results:
        print('\n', 'Sound', '  : ', sig_name[i], file=f)
        i=i+1
        for key, value in res.items():
            print('\n', key, '  : ', value, file=f)


