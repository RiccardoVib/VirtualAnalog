import tensorboard
#load_ext tensorboard
#rm -rf ./logs/
import datetime
import numpy as np
import os
import time
import tensorflow as tf
from TrainFunctionality import coefficient_of_determination
from GetData import get_data
from scipy.io import wavfile
from tensorflow.keras.layers import Input, Dense, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD

#
def trainLSTM(data_dir, epochs, seed=422, data=None, **kwargs):
    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 16)
    learning_rate = kwargs.get('learning_rate', 0.001)
    units = kwargs.get('units', [1])
    model_save_dir = kwargs.get('model_save_dir', '../../LSTM_TrainedModels')
    save_folder = kwargs.get('save_folder', 'LSTM_testing')
    generate_wav = kwargs.get('generate_wav', None)
    drop = kwargs.get('drop', 0.)
    opt_type = kwargs.get('opt_type', 'Adam')
    inference = kwargs.get('inference', False)
    loss_type = kwargs.get('loss_type', 'mae')
    shuffle_data = kwargs.get('shuffle_data', False)
    w_length = kwargs.get('w_length', 0.001)
    n_record = kwargs.get('n_record', 1)

    if data is None:
        x, y, x_val, y_val, x_test, y_test, scaler, zero_value = get_data(data_dir, n_record=n_record, shuffle=shuffle_data, w_length=w_length, seed=seed)
    else:
        x, y, x_val, y_val, x_test, y_test, scaler, zero_value = data

    layers = len(units)
    n_units = ''
    for unit in units:
        n_units += str(unit)+', '

    n_units = n_units[:-2]

    #T past values used to predict the next value
    T = x.shape[1] #time window
    D = x.shape[2]

    inputs = Input(shape=(T,D), name='enc_input')
    first_unit_encoder = units.pop(0)
    if len(units) > 0:
        last_unit_encoder = units.pop()
        outputs = LSTM(first_unit_encoder, return_sequences=True, name='LSTM_En0')(inputs)
        for i, unit in enumerate(units):
            outputs = LSTM(unit, return_sequences=True, name='LSTM_En' + str(i + 1))(outputs)
        outputs, state_h, state_c = LSTM(last_unit_encoder, return_state=True, name='LSTM_EnFin')(outputs)
    else:
        outputs, state_h, state_c = LSTM(first_unit_encoder, return_state=True, name='LSTM_En')(inputs)

    #encoder_states = [state_h, state_c]
    if drop != 0.:
        outputs = tf.keras.layers.Dropout(drop, name='DropLayer')(outputs)
    outputs = Dense(T, activation='sigmoid', name='DenseLay')(outputs)
    model = Model(inputs, outputs)
    model.summary()

    if opt_type == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif opt_type == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError('Please pass opt_type as either Adam or SGD')

    if loss_type == 'mae':
        model.compile(loss='mae', metrics=['mae'], optimizer=opt)
    elif loss_type == 'mse':
        model.compile(loss='mse', metrics=['mse'], optimizer=opt)
    else:
        raise ValueError('Please pass loss_type as either MAE or MSE')

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

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    #train the RNN
    results = model.fit(x, y, batch_size=b_size, epochs=epochs,
                        validation_data=(x_val, y_val),
                        #callbacks=tensorboard_callback)
                        callbacks=callbacks)

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
    predictions_test = model.predict(x_test, batch_size=b_size)

    final_model_test_loss = model.evaluate(x_test, y_test, batch_size=b_size, verbose=0)
    y_s = np.reshape(y_test, (-1))
    y_pred = np.reshape(predictions_test,(-1))
    r_squared = coefficient_of_determination(y_s[:1600], y_pred[:1600])

    if ckpt_flag:
        best = tf.train.latest_checkpoint(ckpt_dir)
        if best is not None:
            print("Restored weights from {}".format(ckpt_dir))
            model.load_weights(best)
    test_loss = model.evaluate(x_test, y_test, batch_size=b_size, verbose=0)
    print('Test Loss: ', test_loss)
    if inference:
        results = {}
    else:
        results = {
            'Test_Loss': test_loss,
            'Min_val_loss': np.min(results.history['val_loss']),
            'Min_train_loss': np.min(results.history['loss']),
            'b_size': b_size,
            'loss_type': loss_type,
            'learning_rate': learning_rate,
            'layers': layers,
            'units':n_units,
            'Train_loss': results.history['loss'],
            'Val_loss': results.history['val_loss'],
            'r_squared': r_squared
        }
        print(results)

    if generate_wav is not None:
        np.random.seed(seed)
        gen_indxs = np.random.choice(len(y_test), generate_wav)
        x_gen = x_test
        y_gen = y_test
        predictions = model.predict(x_gen)
        print('GenerateWavLoss: ', model.evaluate(x_gen, y_gen, batch_size=b_size, verbose=0))
        predictions = scaler[0].inverse_transform(predictions)
        x_gen = scaler[0].inverse_transform(x_gen[:, :, 0])
        y_gen = scaler[0].inverse_transform(y_gen)

        predictions = predictions.reshape(-1)
        x_gen = x_gen.reshape(-1)
        y_gen = y_gen.reshape(-1)

        for i, indx in enumerate(gen_indxs):
            # Define directories
            pred_name = 'LSTM_pred.wav'
            inp_name = 'LSTM_inp.wav'
            tar_name = 'LSTM_tar.wav'

            pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
            inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
            tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

            if not os.path.exists(os.path.dirname(pred_dir)):
                os.makedirs(os.path.dirname(pred_dir))

            # Save Wav files
            predictions = predictions.astype('int16')
            x_gen = x_gen.astype('int16')
            y_gen = y_gen.astype('int16')
            wavfile.write(pred_dir, 16000, predictions)
            wavfile.write(inp_dir, 16000, x_gen)
            wavfile.write(tar_dir, 16000, y_gen)

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
    data_dir = '../Files'
    seed = 422
    #start = time.time()
    trainLSTM(data_dir=data_dir,
              model_save_dir='../../TrainedModels',
              save_folder='LSTM_Testing',
              ckpt_flag=True,
              b_size=128,
              units=[1, 8],
              learning_rate=0.0001,
              epochs=1,
              loss_type='mse',
              generate_wav=2,
              n_record=1,
              w_length=0.001,
              shuffle_data=False)
    #end = time.time()
    #print(end - start)