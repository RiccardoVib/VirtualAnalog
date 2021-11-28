import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from Code.GetData import get_data

from tensorflow.keras.layers import Input, Dense, Flatten, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD


def trainLSTM(data_dir, epochs, seed=422, data=None, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 28)
    learning_rate = kwargs.get('learning_rate', 0.001)
    encoder_units = kwargs.get('encoder_units', [8, 8])
    decoder_units = kwargs.get('decoder_units', [8, 8])
    if encoder_units[-1] != decoder_units[0]:
        raise ValueError('Final encoder layer must same units as first decoder layer!')
    dff_output = kwargs.get('dff_output', 128)
    model_save_dir = kwargs.get('model_save_dir', '../../TrainedModels')
    save_folder = kwargs.get('save_folder', 'Longformer_TESTING')
    generate_wav = kwargs.get('generate_wav', None)


    if data is None:
        x, y, x_val, y_val, scaler, zero_value = get_data(data_dir, batch_size=b_size, seed=seed)
    else:
        x, y, x_val, y_val, scaler, zero_value = data

    #T past values used to predict the next value
    T = x.shape[1]#//2 #time window
    D = 1

    encoder_inputs = Input(shape=(T, D), batch_size=b_size, name='enc_input')
    first_unit_encoder = encoder_units.pop(0)
    if len(encoder_units) > 0:
        last_unit_encoder = encoder_units.pop()
        outputs = LSTM(first_unit_encoder, return_sequences=True, name='LSTM_En1')(encoder_inputs)
        for i, unit in enumerate(encoder_units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_En' + str(i + 1))(outputs)
        outputs, state_h, state_c = LSTM(last_unit_encoder, return_state=True, name='LSTM_EnFin')(outputs)
    else:
        outputs, state_h, state_c = LSTM(first_unit_encoder, return_state=True, name='LSTM_En')(encoder_inputs)

    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(T-1, D), batch_size=b_size, name='dec_input')
    first_unit_decoder = decoder_units.pop(0)
    if len(decoder_units) > 0:
        last_unit_decoder = decoder_units.pop()
        outputs = LSTM(first_unit_decoder, return_sequences=True, name='LSTM_De1')(decoder_inputs,
                                                                                   initial_state=encoder_states)
        for i, unit in enumerate(decoder_units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_De' + str(i + 1))(outputs)
        outputs, _, _ = LSTM(last_unit_decoder, return_sequences=True, return_state=True, name='LSTM_DeFin')(outputs)
    else:
        outputs, _, _ = LSTM(first_unit_decoder, return_sequences=True, return_state=True, name='LSTM_De')(
                                                                                        decoder_inputs,
                                                                                        initial_state=encoder_states)
    outputs = Dense(dff_output, activation='relu', name='Dff_Lay')(outputs)
    decoder_outputs = Dense(D, activation='sigmoid', name='DenseLay')(outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()
    model.compile(loss='mae', metrics=['mae'], optimizer=Adam(learning_rate=learning_rate))

    # TODO: Currently not loading weights as we only save the best model... Should probably
    callbacks = []
    if ckpt_flag:
        ckpt_path = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints', 'cp-{epoch:04d}.ckpt'))
        ckpt_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'Checkpoints'))
        if not os.path.exists(os.path.dirname(ckpt_dir)):
            os.makedirs(os.path.dirname(ckpt_dir))

        ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path, monitor='val_loss', mode='min',
                                                           save_best_only=True, save_weights_only=True, verbose=1, )
        callbacks += [ckpt_callback]
        latest = tf.train.latest_checkpoint(ckpt_dir)
        if latest is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(latest)
            start_epoch = int(latest.split('-')[-1].split('.')[0])
            print('Starting from epoch: ', start_epoch + 1)
        else:
            print("Initializing random weights.")

    #train the RNN
    results = model.fit([x, y[:,:-1]], y[:,1:], batch_size=b_size, epochs=epochs,
                  validation_data=([x_val, y_val[:,:-1]], y_val[:,1:]), callbacks=callbacks)

    #predictions_test = model.predict([X_test, Y_test[:,:-1,:]], batch_size=b_size) #

    #final_model_test_loss = model.evaluate([X_test, Y_test[:, :-1, :]], Y_test[:, 1:, :], batch_size=b_size, verbose=0)
    # if ckpt_flag:
    #     best = tf.train.latest_checkpoint(ckpt_dir)
    #     if best is not None:
    #         print("Restored weights from {}".format(ckpt_dir))
    #         model.load_weights(best)
    # test_loss = model.evaluate([X_test, Y_test[:, :-1, :]], Y_test[:, 1:, :], batch_size=b_size, verbose=0)
    # print('Test Loss: ', test_loss)

    results = {
        #'Test_Loss': test_loss,
        'Min_val_loss': np.min(results.history['val_loss']),
        'Min_train_loss': np.min(results.history['loss']),
        'b_size': b_size,
        'learning_rate': learning_rate,
        'encoder_units': encoder_units,
        'decoder_units': decoder_units,
        'dff_output': dff_output,
        'Train_loss': results.history['loss'],
        'Val_loss': results.history['val_loss']
    }

    # if generate_wav is not None:
    #     np.random.seed(seed)
    #     gen_indxs = np.random.choice(len(y_test), generate_wav)
    #     x_gen = Z[x_test[gen_indxs,0]]
    #     y_gen = Z[y_test[gen_indxs]]
    #     predictions = model.predict([x_gen, y_gen[:,:-1,:]])
    #     print('GenerateWavLoss: ', model.evaluate([x_gen, y_gen[:,:-1,:]], y_gen[:,1:,:], batch_size=b_size, verbose=0))
    #     predictions = scaler.inverse_transform(predictions)
    #     x_gen = scaler.inverse_transform(x_gen)
    #     y_gen = scaler.inverse_transform(y_gen)
    #     for i, indx in enumerate(gen_indxs):
    #         # Define directories
    #         pred_name = 'x' + str(x_test[gen_indxs][i][0]) + '_y' + str(y_test[gen_indxs][i]) + '_pred.wav'
    #         inp_name = 'x' + str(x_test[gen_indxs][i][0]) + '_y' + str(y_test[gen_indxs][i]) + '_inp.wav'
    #         tar_name = 'x' + str(x_test[gen_indxs][i][0]) + '_y' + str(y_test[gen_indxs][i]) + '_tar.wav'
    #
    #         pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
    #         inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
    #         tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))
    #
    #         if not os.path.exists(os.path.dirname(pred_dir)):
    #             os.makedirs(os.path.dirname(pred_dir))
    #
    #         # Save some Spectral Plots:
    #         spectral_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'SpectralPlots'))
    #         if not os.path.exists(spectral_dir):
    #             os.makedirs(spectral_dir)
    #         plot_spectral(Zxx=predictions[i], title='Predictions',
    #                       save_dir=os.path.normpath(os.path.join(spectral_dir, pred_name)).replace('.wav', '.png'))
    #         plot_spectral(Zxx=x_gen[i], title='Inputs',
    #                       save_dir=os.path.normpath(os.path.join(spectral_dir, inp_name)).replace('.wav', '.png'))
    #         plot_spectral(Zxx=y_gen[i], title='Target',
    #                       save_dir=os.path.normpath(os.path.join(spectral_dir, tar_name)).replace('.wav', '.png'))
    #
    #         # Save Wav files
    #         #sp.io.wavfile.write(pred_dir, 44100, pred_i)
    #         #sp.io.wavfile.write(inp_dir, 44100, inp_i)
    #         #sp.io.wavfile.write(tar_dir, 44100, tar_i)


if __name__ == '__main__':
    data_dir = '../../../Data/bach_multiinstrumental/bach_multiinstrumental/Training'
    # data_dir = '/Users/riccardosimionato/Datasets/bach_multiinstrumental'
    seed = 422
    data = get_data(data_dir=data_dir, seed=seed)
    trainLSTM(data_dir=data_dir,
              model_save_dir='../../../TrainedModels',
              save_folder='LSTM_Testing',
              ckpt_flag=True,
              b_size=1,
              learning_rate=0.0001,
              encoder_units=[4, 4],
              decoder_units=[4, 4],
              dff_output=512,
              epochs=1,
              data=data,
              generate_wav=2)