from ProperEvaluationAllModels import load_audio, prediction_accuracy, measure_performance, measure_time, load_model_dense, load_model_lstm
from ProperEvaluationAllModels import load_model_lstm_enc_dec_v2, inferenceLSTM_enc_dec_v2
from ProperEvaluationAllModels import plot_time, plot_fft, create_ref, spectrogram, load_ref
import os
import pickle
import glob
import audio_format
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import fft
from mag_smoothing import mag_smoothing
from sklearn import metrics

def retrive_info(architecture, model_dir, units, drop, w):

    data_ = '../Files'
    file_data = open(os.path.normpath('/'.join([data_, 'data_never_seen_w1.pickle'])), 'rb')
    data = pickle.load(file_data)
    x_test = data['x']
    y_test = data['y']
    fs = 48000
    scaler = data['scaler']

    # Dense-----------------------------------------------------------------------------------
    if architecture=='dense':
        dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/Dense_trials/'
        data_dir = os.path.normpath(os.path.join(dir, model_dir))
        #data_dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/Dense_trials/DenseFeed_Testing_64_in1'
        name = 'Dense'
        T=x_test.shape[1]

        model = load_model_dense(T, units, drop, model_save_dir=data_dir)
        predictions = model.predict(x_test)
        predictions = scaler[0].inverse_transform(predictions)
        x_gen = scaler[0].inverse_transform(x_test[:, :, 0])
        y_gen = scaler[0].inverse_transform(y_test)

        predictions = predictions.reshape(-1)
        x_gen = x_gen.reshape(-1)
        y_gen = y_gen.reshape(-1)

        # Define directories
        pred_name = 'Dense_Loop_pred.wav'
        inp_name = 'Dense_Loop_inp.wav'
        tar_name = 'Dense_Loop_tar.wav'

        pred_dir = os.path.normpath(os.path.join(data_dir, pred_name))
        inp_dir = os.path.normpath(os.path.join(data_dir, inp_name))
        tar_dir = os.path.normpath(os.path.join(data_dir, tar_name))

        predictions = predictions.astype('int16')
        x_gen = x_gen.astype('int16')
        y_gen = y_gen.astype('int16')
        wavfile.write(pred_dir, fs, predictions)
        wavfile.write(inp_dir, fs, x_gen)
        wavfile.write(tar_dir, fs, y_gen)

    # LSTM-----------------------------------------------------------------------------------
    if architecture == 'lstm':
        dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_trials/'
        data_dir = os.path.normpath(os.path.join(dir, model_dir))
        #data_dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_trials/LSTM_Testing_64h'
        name = 'LSTM'
        T=x_test.shape[1]
        model = load_model_lstm(T, units, drop, model_save_dir=data_dir)
        predictions = model.predict(x_test)

        predictions = scaler[0].inverse_transform(predictions)
        x_gen = scaler[0].inverse_transform(x_test[:, :, 0])
        y_gen = scaler[0].inverse_transform(y_test)

        predictions = predictions.reshape(-1)
        x_gen = x_gen.reshape(-1)
        y_gen = y_gen.reshape(-1)

        # Define directories
        pred_name = 'LSTM_Loop_pred.wav'
        inp_name = 'LSTM_Loop_inp.wav'
        tar_name = 'LSTM_Loop_tar.wav'

        pred_dir = os.path.normpath(os.path.join(data_dir, pred_name))
        inp_dir = os.path.normpath(os.path.join(data_dir, inp_name))
        tar_dir = os.path.normpath(os.path.join(data_dir, tar_name))

        predictions = predictions.astype('int16')
        x_gen = x_gen.astype('int16')
        y_gen = y_gen.astype('int16')
        wavfile.write(pred_dir, fs, predictions)
        wavfile.write(inp_dir, fs, x_gen)
        wavfile.write(tar_dir, fs, y_gen)

    # --------------------------------------------------------------------------------------
    # change of dataset
    # --------------------------------------------------------------------------------------

    file_data = open(os.path.normpath('/'.join([data_, 'data_never_seen_w16.pickle'])), 'rb')

    data = pickle.load(file_data)
    x_test = data['x']
    fs = data['fs']
    scaler = data['scaler']
    del data

    # LSTM_enc_dec_v2-------------------------------------------------------------------------
    if architecture == 'lstm_enc_dec_v2':
        dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_enc_dec_v2_trials/'
        data_dir = os.path.normpath(os.path.join(dir, model_dir))
        data_dir_ref = '/Users/riccardosimionato/PycharmProjects/All_Results'
        #data_dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_enc_dec_v2_trials/LSTM_enc_dec_v2_2'
        name = 'LSTM_enc_dec_v2'
        T = x_test.shape[1]
        enc_units = [units[0]]
        dec_units = [units[1]]

        model = load_model_lstm_enc_dec_v2(T=T, encoder_units=enc_units, decoder_units=dec_units, drop=drop, model_save_dir=data_dir)


        inferenceLSTM_enc_dec_v2(data_dir=data_dir, model=model, fs=fs, scaler=scaler, start=0, stop=x_test.shape[0], T=T, name='loop', generate=True)

        # all_results = []
        # for l in range(len(sig_name)):
        #     file_inp = glob.glob(os.path.normpath('/'.join([data_dir_ref, sig_name[l] + '_inp.wav'])))
        #     file_tar = glob.glob(os.path.normpath('/'.join([data_dir_ref, sig_name[l] + '_tar.wav'])))
        #     file_pred = glob.glob(os.path.normpath('/'.join([data_dir, sig_name[l] + '_pred.wav'])))
        #     for file in file_inp:
        #         fs, audio_inp = wavfile.read(file)
        #     for file in file_tar:
        #         fs, audio_tar = wavfile.read(file)
        #     for file in file_pred:
        #         _, audio_pred = wavfile.read(file)
        #
        #     audio_inp = audio_format.pcm2float(audio_inp)
        #     audio_tar = audio_format.pcm2float(audio_tar)
        #     audio_inp = audio_inp[w:]
        #     audio_tar = audio_tar[w:]
        #     audio_pred = audio_format.pcm2float(audio_pred)
        #     plot_time(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM_enc_dec_v2' + sig_name[l])
        #     plot_fft(audio_tar, audio_pred, audio_inp, fs, data_dir, 'LSTM_enc_dec_v2' + sig_name[l])
        #     results = measure_performance(audio_tar, audio_pred, name)
        #     all_results.append(results)

            #spectrogram(audio_tar, audio_pred, audio_inp, fs, data_dir, sig_name[l] + name)


if __name__ == '__main__':

    #retrive_info(architecture='dense', model_dir='DenseFeed_32_32', units=[32, 32], drop=0., w=1)
    #retrive_info(architecture='lstm', model_dir='LSTM_32_32', units=[32, 32], drop=0., w=1)
    #retrive_info(architecture='lstm_enc_dec_v2', model_dir='LSTM_enc_dec_v2_16_64_64', units=[64, 64], drop=0., w=16)

    dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/Loop/'
    file_inp = glob.glob(os.path.normpath('/'.join([dir, 'Loop_inp.wav'])))
    file_tar = glob.glob(os.path.normpath('/'.join([dir, 'Loop_tar.wav'])))
    file_dense_pred = glob.glob(os.path.normpath('/'.join([dir, 'Dense_Loop_pred.wav'])))
    file_lstm_pred = glob.glob(os.path.normpath('/'.join([dir, 'LSTM_Loop_pred.wav'])))
    file_v2_pred = glob.glob(os.path.normpath('/'.join([dir, 'LSTM_v2_loop_pred.wav'])))

    for file in file_inp:
        fs, audio_inp = wavfile.read(file)
    for file in file_tar:
        _, audio_tar = wavfile.read(file)
    for file in file_dense_pred:
        _, audio_pred = wavfile.read(file)

    start = fs
    stop = start + 5*fs
    spectrogram(audio_tar[start:stop], audio_pred[start:stop], audio_inp[start:stop], fs, dir, 'dense')


    for file in file_lstm_pred:
        _, audio_pred = wavfile.read(file)

    spectrogram(audio_tar[start:stop], audio_pred[start:stop], audio_inp[start:stop], fs, dir, 'lstm')

    for file in file_v2_pred:
        _, audio_pred = wavfile.read(file)

    spectrogram(audio_tar[start:stop], audio_pred[start:stop], audio_inp[start:stop], fs, dir, 'v2')