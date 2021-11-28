from Code.Wavenet.WaveNet import WaveNet
import numpy as np
import os
import pickle
import tensorflow as tf
from Code.GetData import get_data
from scipy.io import wavfile
from scipy import signal
from tensorflow.keras.utils import Progbar
from Code.TrainFunctionality import get_batches, PlotLossesSame
from Code.TrainRAMT import plot_spectral, threshold_loss


def trainWavenet(data_dir, epochs, seed=422, data=None, **kwargs):

    if data is None:
        x, y, x_val, y_val, x_test, y_test, Z, scaler, zero_value, indeces, name = get_data(data_dir=data_dir,
                                                                                            seed=seed)
    else:
        x, y, x_val, y_val, x_test, y_test, Z, scaler, zero_value, indeces, name = data

    # -----------------------------------------------------------------------------------------------------------------
    # Set-up model, optimiser, lr_sched and losses:
    # -----------------------------------------------------------------------------------------------------------------
    model_save_dir = kwargs.get('model_save_dir', '/Users/riccardosimionato/Datasets/bach_multiinstrumental')
    #model_save_dir = kwargs.get('model_save_dir', '/mnt')
    save_folder = kwargs.get('save_folder', 'WaveNet')
    generate_wav = kwargs.get('generate_wav', None)
    ckpt_flag = kwargs.get('ckpt_flag', False)
    plot_progress = kwargs.get('plot_progress', True)
    max_length = Z.shape[1]
    learning_rate = kwargs.get('learning_rate', None)
    b_size = kwargs.get('b_size', 16)
    inference_flag = kwargs.get('inference_flag', False)

    output_dim = Z.shape[-1]

    kernel_size = kwargs.get('kernel_size', [1, 3, 1, 1])
    dilation = kwargs.get('dilation', 2)
    residual_channels = kwargs.get('residual_channels', 128)
    gate_channels = kwargs.get('gate_channels', 256)
    filters = kwargs.get('filters', [128, 128, 513])

    ckpt_flag = kwargs.get('ckpt_flag', False)
    b_size = kwargs.get('b_size', 1)
    data = get_data(data_dir=data_dir, seed=seed)
    epochs = kwargs.get('epochs', 1)

    #learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.1, decay_steps=100000, decay_rate=0.96)
    opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    wavenet = WaveNet(dilation, kernel_size, filters, residual_channels, gate_channels)

    loss_fn = tf.keras.losses.MeanAbsoluteError()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_loss_thres = tf.keras.metrics.Mean(name='val_loss_thres')
    val_loss_mae = tf.keras.metrics.Mean(name='val_loss_mae')
    val_loss_sthres = tf.keras.metrics.Mean(name='val_loss_sthres')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_loss_thres = tf.keras.metrics.Mean(name='test_loss_thres')
    test_loss_mae = tf.keras.metrics.Mean(name='test_loss_mae')
    test_loss_sthres = tf.keras.metrics.Mean(name='test_loss_sthres')

    # -----------------------------------------------------------------------------------------------------------------
    # Define the training functionality
    # -----------------------------------------------------------------------------------------------------------------

    @tf.function
    def train_step(inp, tar):
        #tar_inp = tar[:, :-1, :]
        #tar_real = tar[:, 1:, :]

        #inp = tf.repeat(904, inp, axis=1)
        #layer = tf.layer.Dense(1, activation=None)
        #new_inp = layer(inp) #(3, 905)

        with tf.GradientTape() as tape:
            #predictions = wavenet(inp[:, :-1, :], tar_inp)
            #predictions = wavenet(tar_inp, inp)
            predictions = wavenet(inp)
            loss_thres, loss_sthres = threshold_loss(tar, predictions, zero_value=0.)
            loss = loss_thres + loss_sthres

        gradients = tape.gradient(loss, wavenet.trainable_variables)
        opt.apply_gradients(zip(gradients, wavenet.trainable_variables))

        train_loss.update_state(loss)

    @tf.function
    def val_step(inp, tar, testing=False):
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]

        #predictions = wavenet(inp[:, :-1, :], tar_inp)
        predictions = wavenet(inp)
        loss_thres, loss_sthres = threshold_loss(tar, predictions, zero_value=0.)
        loss = loss_thres + loss_sthres
        loss_mae = loss_fn(tar, predictions)

        val_loss.update_state(loss)

        if not testing:
            val_loss.update_state(loss)
            val_loss_thres.update_state(loss_thres)
            val_loss_sthres.update_state(loss_sthres)
            val_loss_mae.update_state(loss_mae)
        else:
            test_loss.update_state(loss)
            test_loss_thres.update_state(loss_thres)
            test_loss_sthres.update_state(loss_sthres)
            test_loss_mae.update_state(loss_mae)

    # -----------------------------------------------------------------------------------------------------------------
    # Set up checkpointing (saving) of the model (and load if present)
    # -----------------------------------------------------------------------------------------------------------------

    start_epoch = 0
    if ckpt_flag or inference_flag:
        Z_shape = tf.shape(Z)
        graph_signature = [
            tf.TensorSpec((None, Z_shape[1], Z_shape[2]), tf.float32),
            tf.TensorSpec((None, Z_shape[1], Z_shape[2]), tf.float32)
        ]

    @tf.function(input_signature=graph_signature)
    def inference(tar, inp):
        outputs = wavenet(inp)
        return outputs

    save_model_latest = os.path.normpath('/'.join([model_save_dir, save_folder, 'Latest']))
    save_model_best = os.path.normpath('/'.join([model_save_dir, save_folder, 'Best']))

    # Load model if dir exists
    if os.path.exists(save_model_latest):
        f = open('/'.join([os.path.dirname(save_model_best), 'epoch.txt']))
        ckpt_info = f.read()
        f.close()
        start_epoch = [int(s) for s in ckpt_info.split() if s.isdigit()][0]  # Get the latest epoch it trained
        print('Loading weights and starting from epoch ', start_epoch)
        if inference_flag:
            loaded = tf.saved_model.load(save_model_best)
        else:
            loaded = tf.saved_model.load(save_model_latest)
        # Need to make a single prediction for the model as it needs to compile:
        wavenet(tf.constant(Z[x[:2]], dtype='float32'))
        for i in range(len(wavenet.variables)):
            if wavenet.variables[i].name != loaded.all_variables[i].name:
                assert ValueError('Cannot load model, due to incompatible loaded and model...')
            wavenet.variables[i].assign(loaded.all_variables[i].value())
    else:
        print('Weights were randomly initialised')

    # Try to load loss figure plot if exists
    if os.path.exists(
            os.path.normpath('/'.join([model_save_dir, save_folder, 'fig_progress.pickle']))) and plot_progress:
        try:
            fig_progress = pickle.load(
                open(os.path.normpath('/'.join([model_save_dir, save_folder, 'fig_progress.pickle'])), 'rb'))
        except (ValueError, Exception):
            print('Could not load loss figure')
            pass


    ckpt_interval = 1

    # -----------------------------------------------------------------------------------------------------------------
    # Train the model
    # -----------------------------------------------------------------------------------------------------------------
    _logs = [[], [], [], [], []]
    min_val_error = np.inf
    summary_res = 1

    if inference_flag:
        start_epoch = epochs

    for epoch in range(start_epoch, epochs):
        train_loss.reset_states()
        val_loss.reset_states()
        val_loss_thres.reset_states()
        val_loss_mae.reset_states()
        val_loss_sthres.reset_states()

        # Get batches
        x_batches, y_batches = get_batches(x, y, b_size=b_size, shuffle=True, seed=epoch)
        # Set-up training progress bar
        n_batch = len(x_batches)
        print("\nepoch {}/{}".format(epoch + 1, epochs))
        pb_i = Progbar(n_batch * b_size, stateful_metrics=['Loss: '])

        for batch_num, (x_batch_i, y_batch_i) in enumerate(zip(x_batches, y_batches)):

            x_batch = tf.constant(Z[x_batch_i], dtype='float32')
            y_batch = tf.constant(Z[y_batch_i], dtype='float32')
            train_step(inp=x_batch, tar=y_batch)
            # Print progbar
            if batch_num % summary_res == 0:
                values = [('Loss: ', train_loss.result())]
                pb_i.add(b_size * summary_res, values=values)

        # -------------------------------------------------------------------------------------------------------------
        # Validate the model
        # -------------------------------------------------------------------------------------------------------------

        # Get batches
        x_batches, y_batches = get_batches(x_val, y_val, b_size=b_size, shuffle=True, seed=epoch)

        for (batch_num, (x_batch_i, y_batch_i)) in enumerate(zip(x_batches, y_batches)):

            x_batch = tf.constant(Z[x_batch_i], dtype='float32')
            y_batch = tf.constant(Z[y_batch_i], dtype='float32')

            val_step(inp=x_batch, tar=y_batch)

        # Print validation losses:
        print('\nValidation Loss:', val_loss.result().numpy())
        print('         Thres: ', val_loss_thres.result().numpy())
        print('     SmalThres: ', val_loss_sthres.result().numpy())
        print('         MAE:   ', val_loss_mae.result().numpy())

        # -------------------------
        # *** Checkpoint Model: ***
        # -------------------------
        if ckpt_flag:
            if (epoch % ckpt_interval == 0) or (val_loss.result() < min_val_error):
                to_save = tf.Module()
                to_save.inference = inference
                to_save.all_variables = list(wavenet.variables)
                tf.saved_model.save(to_save, save_model_latest)

                if val_loss.result() < min_val_error:
                    print('*** New Best Model Saved to %s ***' % save_model_best)
                    to_save = tf.Module()
                    to_save.inference = inference
                    to_save.all_variables = list(wavenet.variables)
                    tf.saved_model.save(to_save, save_model_best)
                    best_epoch = epoch

                epoch_dir = '/'.join([os.path.dirname(save_model_best), 'epoch.txt'])
                f = open(epoch_dir, 'w+')
                f.write('Latest Epoch: %s \nBest Epoch: %s \n' % (epoch + 1, best_epoch + 1))
                f.close()

        # -----------------------------
        # *** Plot Training Losses: ***
        # -----------------------------
        if plot_progress:
            if 'fig_progress' not in locals():
                fig_progress = PlotLossesSame(epoch + 1,
                                                Training=train_loss.result().numpy(),
                                                Validation=val_loss.result().numpy(),
                                                Val_Thres=val_loss_thres.result().numpy(),
                                                Val_sThres=val_loss_sthres.result().numpy(),
                                                Val_MAE=val_loss_mae.result().numpy())
            else:
                fig_progress.on_epoch_end(Training=train_loss.result().numpy(),
                                          Validation=val_loss.result().numpy(),
                                          Val_Thres=val_loss_thres.result().numpy(),
                                          Val_sThres=val_loss_sthres.result().numpy(),
                                          Val_MAE=val_loss_mae.result().numpy())
            # Store the plot if ckpting:
            if ckpt_flag:
                fig_progress.fig.savefig(os.path.normpath('/'.join([model_save_dir, save_folder, 'val_loss.png'])))
                if not os.path.exists(os.path.normpath('/'.join([model_save_dir, save_folder]))):
                    os.makedirs(os.path.normpath('/'.join([model_save_dir, save_folder])))
                file_fig = open(os.path.normpath('/'.join([model_save_dir, save_folder, 'fig_progress.pickle'])), 'wb')
                pickle.dump(fig_progress, file_fig)
                #pickle.dump(fig_progress, os.path.normpath('/'.join([model_save_dir, save_folder, 'fig_progress.pickle'])))

        # Store currently best validation loss:
        if val_loss.result() < min_val_error:
            min_val_error = val_loss.result()

        # Append epoch losses to logs:
        _logs[0].append(train_loss.result().numpy())
        _logs[1].append(val_loss.result().numpy())
        _logs[2].append(val_loss_thres.result().numpy())
        _logs[3].append(val_loss_sthres.result().numpy())
        _logs[4].append(val_loss_mae.result().numpy())

        if epoch == start_epoch:
            n_params = np.sum([np.prod(v.get_shape()) for v in wavenet.variables])
            print('Number of parameters: ', n_params)

    # -----------------------------------------------------------------------------------------------------------------
    # Test the model
    # -----------------------------------------------------------------------------------------------------------------
    # # Load the best model:
    if ckpt_flag and not inference_flag:
        if os.path.exists(save_model_best):
            f = open('/'.join([os.path.dirname(save_model_best), 'epoch.txt']))
            ckpt_info = f.read()
            f.close()
            start_epoch = [int(s) for s in ckpt_info.split() if s.isdigit()][1]  # Get the latest epoch it trained
            print('Loading weights from best epoch ', start_epoch)
            loaded = tf.saved_model.load(save_model_best)

            # Need to make a single prediction for the model as it needs to compile:
            wavenet(tf.constant(Z[x[:2]], dtype='float32'))
            for i in range(len(wavenet.variables)):
                if wavenet.variables[i].name != loaded.all_variables[i].name:
                    assert ValueError('Cannot load model, due to incompatible loaded and model...')
                wavenet.variables[i].assign(loaded.all_variables[i].value())

    if 'n_params' not in locals():
        n_params = np.sum([np.prod(v.get_shape()) for v in wavenet.variables])
    if 'epoch' not in locals():
        _logs = [[0]]*len(_logs)
        epoch = 0



    # Get batches
    x_batches, y_batches = get_batches(x_test, y_test, b_size=b_size, shuffle=False)
    for (batch_num, (x_batch_i, y_batch_i)) in enumerate(zip(x_batches, y_batches)):
        x_batch = Z[x_batch_i]
        y_batch = Z[y_batch_i]

        x_batch = tf.constant(x_batch, dtype='float32')
        y_batch = tf.constant(y_batch, dtype='float32')

        val_step(inp=x_batch, tar=y_batch, testing=True)

    print('\n\nTest Loss: ', test_loss.result().numpy())
    print('    Thres: ', test_loss_thres.result().numpy())
    print('  SmThres: ', test_loss_sthres.result().numpy())
    print('    MAE:   ', test_loss_mae.result().numpy(), '\n\n')


    if inference_flag:
        results = {}
    else:
        results = {
            'Test_Loss': test_loss.result().numpy(),
            'Test_Loss_Thres': test_loss_thres.result().numpy(),
            'Test_Loss_MAE': test_loss_mae.result().numpy(),
            'b_size': b_size,
            'kernel_size': kernel_size,
            'filters' : filters,
            'residual_channels' : residual_channels,
            'gate_channels' : gate_channels,
            'dilation': dilation,
            'n_params': n_params,
            'learning_rate': learning_rate if isinstance(learning_rate, float) else 'Sched',
            'min_val_loss': np.min(_logs[1]),
            'min_train_loss': np.min(_logs[0]),
            'val_loss': _logs[1],
            'val_loss_thres': _logs[2],
            'val_loss_sthres': _logs[3],
            'val_loss_mae': _logs[4],
            'train_loss': _logs[0],
        }


    if ckpt_flag and not inference_flag:
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)

    # -----------------------------------------------------------------------------------------------------------------
    # Save some Wav-file Predictions (from test set):
    # -----------------------------------------------------------------------------------------------------------------

    if generate_wav is not None:
        gen_indxs = np.random.choice(len(y_test), generate_wav)
        x_gen = Z[x_test[gen_indxs]]
        y_gen = Z[y_test[gen_indxs]]
        predictions = wavenet(tf.constant(x_gen, dtype='float32'))
        predictions = predictions.numpy()
        losses = [loss_fn(lab, pred).numpy() for (lab, pred) in zip(y_gen, predictions)]
        predictions = scaler.inverse_transform(predictions)
        x_gen = scaler.inverse_transform(x_gen)
        y_gen = scaler.inverse_transform(y_gen)
        for i, indx in enumerate(gen_indxs):
            # Define directories
            pred_name = 'x' + str(x_test[gen_indxs][i]) + '_y' + str(y_test[gen_indxs][i]) + '_pred.wav'
            inp_name = 'x' + str(x_test[gen_indxs][i]) + '_y' + str(y_test[gen_indxs][i]) + '_inp.wav'
            tar_name = 'x' + str(x_test[gen_indxs][i]) + '_y' + str(y_test[gen_indxs][i]) + '_tar.wav'

            pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
            inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
            tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

            if not os.path.exists(os.path.dirname(pred_dir)):
                os.makedirs(os.path.dirname(pred_dir))

            # Save some Spectral Plots:
            spectral_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'SpectralPlots'))
            if not os.path.exists(spectral_dir):
                os.makedirs(spectral_dir)
            plot_spectral(Zxx=predictions[i], title='Predictions',
                          save_dir=os.path.normpath(os.path.join(spectral_dir, pred_name)).replace('.wav', '.png'))
            plot_spectral(Zxx=x_gen[i], title='Inputs',
                          save_dir=os.path.normpath(os.path.join(spectral_dir, inp_name)).replace('.wav', '.png'))
            plot_spectral(Zxx=y_gen[i], title='Target',
                          save_dir=os.path.normpath(os.path.join(spectral_dir, tar_name)).replace('.wav', '.png'))

            # Inverse STFT
            _, pred_i = signal.istft(predictions[i].T, nperseg=indeces['nperseg_i'], nfft=indeces['nfft'])
            _, inp_i = signal.istft(x_gen[i].T, nperseg=indeces['nperseg_i'], nfft=indeces['nfft'])
            _, tar_i = signal.istft(y_gen[i].T, nperseg=indeces['nperseg_i'], nfft=indeces['nfft'])

            # Resample
            pred_i = signal.resample_poly(pred_i, up=44100 // indeces['samplerate'], down=1)
            pred_i = pred_i.astype('int16')
            inp_i = signal.resample_poly(inp_i, up=44100 // indeces['samplerate'], down=1)
            inp_i = inp_i.astype('int16')
            tar_i = signal.resample_poly(tar_i, up=44100 // indeces['samplerate'], down=1)
            tar_i = tar_i.astype('int16')

            # Save Wav files
            wavfile.write(pred_dir, 44100, pred_i)
            wavfile.write(inp_dir, 44100, inp_i)
            wavfile.write(tar_dir, 44100, tar_i)

    return results


if __name__ == '__main__':
    data_dir = '/Users/riccardosimionato/Datasets/bach_multiinstrumental'
    #data_dir = '/mnt/riccardo/'
    seed = 422
    data = get_data(data_dir, seed=422)
    trainWavenet(data_dir=data_dir,
                 model_save_dir='../../TrainedModels',
                 save_folder='Wavenet_1',
                 ckpt_flag=False,
                 plot_progress=True,
                 epochs=4000,
                 generate_wav=10,
                 dilation_rate=2,
                 kernel_size=[1, 3, 1, 1],
                 filters=[128, 128, 513],
                 residual_channels=128,
                 gate_channels=256,
                 b_size=16,
                 learning_rate=0.0001,
                 inference_flag=True)
