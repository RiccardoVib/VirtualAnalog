import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from Code.GetData import get_data
from scipy.io import wavfile
from scipy import signal
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD


def trainDense(data_dir, epochs, seed=422, data=None, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 16)
    learning_rate = kwargs.get('learning_rate', 0.001)
    encoder_units = kwargs.get('encoder_units', [8, 8])
    decoder_units = kwargs.get('decoder_units', [8, 8])
    if encoder_units[-1] != decoder_units[0]:
        raise ValueError('Final encoder layer must same units as first decoder layer!')
    dff_output = kwargs.get('dff_output', 128)
    model_save_dir = kwargs.get('model_save_dir', '../../Dense_TrainedModels')
    save_folder = kwargs.get('save_folder', 'Dense_TESTING')
    generate_wav = kwargs.get('generate_wav', None)
    drop = kwargs.get('drop', 0.)
    opt_type = kwargs.get('opt_type', 'Adam')
    inference = kwargs.get('inference', False)

    if data is None:
        x, y, x_val, y_val, x_test, y_test, r_train, r_val, r_test, scaler, zero_value = get_data(data_dir, batch_size=b_size, seed=seed)
    else:
        x, y, x_val, y_val, x_test, y_test, r_train, r_val, r_test, scaler, zero_value = data

    #T past values used to predict the next value
    T = x.shape[1]#//2 #time window
    D = 1
    C = 1
    out_D = x.shape[1]-1

    encoder_inputs = Input(shape=(T,D), name='enc_input')
    first_unit_encoder = encoder_units.pop(0)
    if len(encoder_units) > 0:
        last_unit_encoder = encoder_units.pop()
        encoder_outputs = Dense(first_unit_encoder, name='Dense_En0')(encoder_inputs)
        for i, unit in enumerate(encoder_units):
            encoder_outputs = Dense(unit, name='Dense_En' + str(i + 1))(encoder_outputs)
        encoder_outputs = Dense(last_unit_encoder, name='Dense_EnFin')(encoder_outputs)
    else:
        encoder_outputs = Dense(first_unit_encoder, name='Dense_En')(encoder_inputs)

    decoder_inputs = Input(shape=(T-1,D), name='dec_input')
    first_unit_decoder = decoder_units.pop(0)
    if len(decoder_units) > 0:
        last_unit_decoder = decoder_units.pop()
        outputs = Dense(first_unit_decoder, name='Dense_De0')(decoder_inputs)
        for i, unit in enumerate(decoder_units):
            outputs = Dense(unit, name='Dense_De' + str(i + 1))(outputs)
        outputs = Dense(last_unit_decoder, name='Dense_DeFin')(outputs)
    else:
        outputs = Dense(first_unit_decoder, name='Dense_De')(decoder_inputs)
    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    #outputs = Dense(dff_output, activation='relu', name='Dff_Lay')(outputs)
    decoder_outputs = Dense(1, activation='sigmoid', name='DenseLay')(outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()

    if opt_type == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_type == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError('Please pass opt_type as either Adam or SGD')

    model.compile(loss='mae', metrics=['mae'], optimizer=opt)

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

    results = model.fit([x, y[:, :-1]], y[:, 1:], batch_size=16, epochs=epochs,
                        validation_data=([x_val, y_val[:, :-1]], y_val[:, 1:]), callbacks=callbacks)

    # #prediction test
    # predictions = []
    # #last train input
    # last_x = x_test[:, :-1]  # DxT array of length T
    #
    # while len(predictions) < len(y_test):
    #     p = model.predict([last_x[0, :], y_test[0, :-1]]) # 1x1 array -> scalar
    #     predictions.append(p)
    #     last_x = np.roll(last_x, -1)
    #
    #     for i in range(last_x.shape[0]):
    #         last_x[-1, i] = p
    #
    #
    # plt.plot(y_test, label='forecast target')
    # plt.plot(predictions, label='forecast prediction')
    # plt.legend()
    predictions_test = model.predict([x_test, y_test[:, :-1]], batch_size=16)

    final_model_test_loss = model.evaluate([x_test, y_test[:, :-1]], y_test[:, 1:], batch_size=b_size, verbose=0)
    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
    test_loss = model.evaluate([x_test, y_test[:, :-1]], y_test[:, 1:], batch_size=b_size, verbose=0)
    print('Test Loss: ', test_loss)
    if inference:
        results = {}
    else:
        results = {
            'Test_Loss': test_loss,
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

    if generate_wav is not None:
        np.random.seed(seed)
        gen_indxs = np.random.choice(len(y_test), generate_wav)
        x_gen = x_test
        y_gen = y_test
        predictions = model.predict([x_gen, y_gen[:, :-1]])
        print('GenerateWavLoss: ', model.evaluate([x_gen, y_gen[:, :-1]], y_gen[:, 1:], batch_size=b_size, verbose=0))
        predictions = scaler[1].inverse_transform(predictions)
        x_gen = scaler[0].inverse_transform(x_gen)
        y_gen = scaler[1].inverse_transform(y_gen)
        predictions = predictions.reshape(-1)
        x_gen = x_gen.reshape(-1)
        y_gen = y_gen.reshape(-1)
        for i, indx in enumerate(gen_indxs):
            # Define directories
            pred_name = '_pred.wav'
            inp_name = '_inp.wav'
            tar_name = '_tar.wav'

            pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
            inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
            tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

            if not os.path.exists(os.path.dirname(pred_dir)):
                os.makedirs(os.path.dirname(pred_dir))

            # Save Wav files
            predictions = predictions.astype('int16')
            x_gen = x_gen.astype('int16')
            y_gen = y_gen.astype('int16')
            wavfile.write(pred_dir, 16000, predictions.T)
            wavfile.write(inp_dir, 16000, x_gen.T)
            wavfile.write(tar_dir, 16000, y_gen.T)

            # Save some Spectral Plots:
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
    #
    return results

if __name__ == '__main__':
    data_dir = '/Users/riccardosimionato/Datasets/VA/VA_results'
    #data_dir = 'C:/Users/riccarsi/Documents/GitHub/VA_pickle'
    seed = 422
    data = get_data(data_dir=data_dir, seed=seed)
    trainDense(data_dir=data_dir,
              model_save_dir='../../../TrainedModels',
              save_folder='Dense_Testing',
              ckpt_flag=True,
              b_size=16,
              learning_rate=0.0001,
              encoder_units=[3, 2],
              decoder_units=[2, 2],
              epochs=1,
              data=data,
              generate_wav=2)