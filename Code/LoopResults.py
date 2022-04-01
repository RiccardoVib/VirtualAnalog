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
import tensorflow as tf

def retrive_info(architecture, model_dir, units, drop, w):

    data_ = '../Files'
    file_data = open(os.path.normpath('/'.join([data_, 'data_never_seen_w1.pickle'])), 'rb')
    #file_data = open(os.path.normpath('/'.join([data_, 'data_prepared_w1_test_samples.pickle'])), 'rb')
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
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', metrics=['mse'], optimizer=opt)

        test_loss = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
        print(test_loss)
        # predictions = model.predict(x_test)
        # predictions = scaler[0].inverse_transform(predictions)
        # x_gen = scaler[0].inverse_transform(x_test[:, :, 0])
        # y_gen = scaler[0].inverse_transform(y_test)
        #
        # predictions = predictions.reshape(-1)
        # x_gen = x_gen.reshape(-1)
        # y_gen = y_gen.reshape(-1)
        #
        # # Define directories
        # pred_name = 'Dense_Loop_pred.wav'
        # inp_name = 'Dense_Loop_inp.wav'
        # tar_name = 'Dense_Loop_tar.wav'
        #
        # pred_dir = os.path.normpath(os.path.join(data_dir, pred_name))
        # inp_dir = os.path.normpath(os.path.join(data_dir, inp_name))
        # tar_dir = os.path.normpath(os.path.join(data_dir, tar_name))
        #
        # predictions = predictions.astype('int16')
        # x_gen = x_gen.astype('int16')
        # y_gen = y_gen.astype('int16')
        # wavfile.write(pred_dir, fs, predictions)
        # wavfile.write(inp_dir, fs, x_gen)
        # wavfile.write(tar_dir, fs, y_gen)

    # LSTM-----------------------------------------------------------------------------------
    if architecture == 'lstm':
        dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_trials/'
        data_dir = os.path.normpath(os.path.join(dir, model_dir))
        #data_dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/LSTM_trials/LSTM_Testing_64h'
        name = 'LSTM'
        T=x_test.shape[1]
        model = load_model_lstm(T, units, drop, model_save_dir=data_dir)
        opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(loss='mse', metrics=['mse'], optimizer=opt)
        test_loss = model.evaluate(x_test, y_test, batch_size=128, verbose=0)
        print(test_loss)
        #
        # predictions = model.predict(x_test)
        #
        # predictions = scaler[0].inverse_transform(predictions)
        # x_gen = scaler[0].inverse_transform(x_test[:, :, 0])
        # y_gen = scaler[0].inverse_transform(y_test)
        #
        # predictions = predictions.reshape(-1)
        # x_gen = x_gen.reshape(-1)
        # y_gen = y_gen.reshape(-1)
        #
        # # Define directories
        # pred_name = 'LSTM_Loop_pred.wav'
        # inp_name = 'LSTM_Loop_inp.wav'
        # tar_name = 'LSTM_Loop_tar.wav'
        #
        # pred_dir = os.path.normpath(os.path.join(data_dir, pred_name))
        # inp_dir = os.path.normpath(os.path.join(data_dir, inp_name))
        # tar_dir = os.path.normpath(os.path.join(data_dir, tar_name))
        #
        # predictions = predictions.astype('int16')
        # x_gen = x_gen.astype('int16')
        # y_gen = y_gen.astype('int16')
        # wavfile.write(pred_dir, fs, predictions)
        # wavfile.write(inp_dir, fs, x_gen)
        # wavfile.write(tar_dir, fs, y_gen)

    # --------------------------------------------------------------------------------------
    # change of dataset
    # --------------------------------------------------------------------------------------

    #file_data = open(os.path.normpath('/'.join([data_, 'data_prepared_w16_test_samples.pickle'])), 'rb') #test samples all settings
    file_data = open(os.path.normpath('/'.join([data_, 'data_never_seen_w16.pickle'])), 'rb') #test samples never seen all settings
    data = pickle.load(file_data)
    x_test = data['x']
    y_test = data['y']
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
        D = x_test.shape[2]
        enc_units = [units[0]]
        dec_units = [units[1]]

        model = load_model_lstm_enc_dec_v2(T=T, D=D, encoder_units=enc_units, decoder_units=dec_units, drop=drop, model_save_dir=data_dir)
        #opt = tf.keras.optimizers.Adam(learning_rate=0.001)
        #model.compile(loss='mse', metrics=['mse'], optimizer=opt)
        #test_loss = model.evaluate([x_test[:, :-1, :], x_test[:, -1, 0].reshape(x_test.shape[0], 1, 1)], y_test[:, -1], batch_size=128, verbose=0)
        #print(test_loss)

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
    # test samples never seen all settings (bass) - #test samples all settings (all)

    #retrive_info(architecture='dense', model_dir='DenseFeed_128_128', units=[128, 128], drop=0., w=1)
    #8 = 0.0006455745315179229 - 0.00028656134963966906
    #16 = 0.0006719152443110943 - 0.00029173449729569256
    #32 = 0.000631364993751049 - 0.000253815611358732
    #64 = 0.0006707076681777835 - 0.0002837239298969507
    #128 = 0.0006627582479268312 - 0.00028080682386644185
    #retrive_info(architecture='lstm', model_dir='LSTM_128_128', units=[128, 128], drop=0., w=1)
    #8 = 0.0006831666687503457 - 0.0002679502358660102
    #16 = 0.0006720478995703161 - 0.00024218247563112527
    #32 = 0.0006786261801607907 - 0.00024300726363435388
    #64 = 0.0006903575267642736 - 0.00024684221716597676
    #128 = 0.0006736059440299869 - 0.00024371933250222355
    retrive_info(architecture='lstm_enc_dec_v2', model_dir='LSTM_enc_dec_v2_mae', units=[64, 64], drop=0., w=16)
    #64 = 0.0012393008219078183 - 0.007002443540841341
    #32 = 0.0012400229461491108 -
    #16 = 0.001235383446328342
    #8 = 0.0012214540038257837

    #2 = 0.01217335369437933 - 0.021829675883054733
    #4 = 0.0021775169298052788 - 0.0062402840703725815
    #8 = 0.0012916452251374722 - 0.006694374606013298
    #16 = 0.0012214540038257837 - 0.007002443540841341

    # dir = '/Users/riccardosimionato/PycharmProjects/TrialsDAFx/Loop/'
    # file_inp = glob.glob(os.path.normpath('/'.join([dir, 'Loop_inp.wav'])))
    # file_tar = glob.glob(os.path.normpath('/'.join([dir, 'Loop_tar.wav'])))
    # file_dense_pred = glob.glob(os.path.normpath('/'.join([dir, 'Dense_Loop_pred.wav'])))
    # file_lstm_pred = glob.glob(os.path.normpath('/'.join([dir, 'LSTM_Loop_pred.wav'])))
    # file_v2_pred = glob.glob(os.path.normpath('/'.join([dir, 'LSTM_v2_loop_pred.wav'])))
    #
    # for file in file_inp:
    #     fs, audio_inp = wavfile.read(file)
    # for file in file_tar:
    #     _, audio_tar = wavfile.read(file)
    # for file in file_dense_pred:
    #     _, audio_pred = wavfile.read(file)
    #
    # start = fs
    # stop = start + 5*fs
    # spectrogram(audio_tar[start:stop], audio_pred[start:stop], audio_inp[start:stop], fs, dir, 'dense')
    #
    #
    # for file in file_lstm_pred:
    #     _, audio_pred = wavfile.read(file)
    #
    # spectrogram(audio_tar[start:stop], audio_pred[start:stop], audio_inp[start:stop], fs, dir, 'lstm')
    #
    # for file in file_v2_pred:
    #     _, audio_pred = wavfile.read(file)
    #
    # spectrogram(audio_tar[start:stop], audio_pred[start:stop], audio_inp[start:stop], fs, dir, 'v2')