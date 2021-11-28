from Code.GetData import get_data
import os
import tensorflow as tf
import numpy as np
#import pandas as pd
import pickle
from scipy.io import wavfile
from scipy import signal
from Code.Models import Transformer, Longformer
from Code.TrainFunctionality import CustomSchedule, get_batches, PlotLossesSame
from tensorflow.keras.utils import Progbar
import matplotlib.pyplot as plt


def plot_spectral(Zxx, title, save_dir=None):
    fig, axs = plt.subplots()
    pcm = axs.pcolormesh(np.arange(Zxx.shape[0]), np.arange(Zxx.shape[-1]), Zxx.T, vmin=0,
                   vmax=np.max(Zxx.T), shading='gouraud')
    fig.colorbar(pcm)
    axs.set_title(title)
    axs.set_ylabel('Frequency [Hz]')
    axs.set_xlabel('Time [sec]')

    if save_dir is not None:
        fig.savefig(save_dir)
    else:
        plt.show()
    plt.close(fig)



def train_RAMT(data_dir, epochs, seed=422, data=None, **kwargs):
    # Get the data:
    if data is None:
        x, y, x_val, y_val, scaler, zero_value = get_data(data_dir, batch_size=b_size, seed=seed)
    else:
        x, y, x_val, y_val, scaler, zero_value = data

    # -----------------------------------------------------------------------------------------------------------------
    # Set-up model, optimiser, lr_sched and losses:
    # -----------------------------------------------------------------------------------------------------------------
    model_save_dir = kwargs.get('model_save_dir', '../../Transformer_TrainedModels')   # TODO: Change
    save_folder = kwargs.get('save_folder', 'Transformer_testing')
    generate_wav = kwargs.get('generate_wav', None)
    ckpt_flag = kwargs.get('ckpt_flag', False)
    plot_progress = kwargs.get('plot_progress', True)
    max_length = x.shape[1]
    learning_rate = kwargs.get('learning_rate', None)
    b_size = kwargs.get('b_size', 28)
    num_layers = kwargs.get('num_layers', 4)
    d_model = kwargs.get('d_model', 128)
    dff = kwargs.get('dff', 512)
    num_heads = kwargs.get('num_heads', 8)
    drop = kwargs.get('drop', .2)
    output_dim = x.shape[-1]
    if learning_rate is None:
        learning_rate = CustomSchedule(d_model=d_model, warmup_steps=4000)
    opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    transformer = Transformer.Transformer(num_layers=num_layers,
                                          d_model=d_model,
                                          num_heads=num_heads,
                                          dff=dff,  # Hidden layer size of feedforward networks
                                          input_vocab_size=None,  # Not relevant for ours as we don't use embedding
                                          target_vocab_size=None,
                                          pe_input=max_length,  # Max length for positional encoding input
                                          pe_target=max_length,
                                          output_dim=output_dim,
                                          rate=drop)  # Dropout rate

    loss_fn = tf.keras.losses.MeanAbsoluteError()       # loss_fn(sampling_rate=8820, a=.5, b=.5)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # -----------------------------------------------------------------------------------------------------------------
    # Define the training functionality
    # -----------------------------------------------------------------------------------------------------------------
    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]

        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp], training=True)

            loss = loss_fn(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        opt.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss.update_state(loss)

    @tf.function
    def val_step(inp, tar, testing=False):
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]

        predictions, attn_weights = transformer([inp, tar_inp], training=False)

        loss = loss_fn(tar_real, predictions)

        if not testing:
            val_loss.update_state(loss)
        else:
            test_loss.update_state(loss)
        return attn_weights

    # -----------------------------------------------------------------------------------------------------------------
    # Set up checkpointing (saving) of the model (and load if present)
    # -----------------------------------------------------------------------------------------------------------------
    start_epoch = 0
    if ckpt_flag:
        Z_shape = tf.shape(x)
        graph_signature = [
            tf.TensorSpec((None, Z_shape[1], Z_shape[2]), tf.float32),
            tf.TensorSpec((None, Z_shape[1], Z_shape[2]), tf.float32)
        ]

        @tf.function(input_signature=graph_signature)
        def inference(tar, inp):
            tar_inp = tar[:, :-1, :]

            outputs = transformer([inp, tar_inp], training=False)
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
            loaded = tf.saved_model.load(save_model_latest)

            # Need to make a single prediction for the model as it needs to compile:
            transformer([tf.costant(x[:2], dtype='float32'),
                         tf.constant(y[:2][:, :-1, :], dtype='float32')],
                        training=False)
            for i in range(len(transformer.variables)):
                if transformer.variables[i].name != loaded.all_variables[i].name:
                    assert ValueError('Cannot load model, due to incompatible loaded and model...')
                transformer.variables[i].assign(loaded.all_variables[i].value())
        else:
            print('Weights were randomly initialised')

        # Try to load loss figure plot if exists
        if os.path.exists(os.path.normpath('/'.join([model_save_dir, save_folder, 'fig_progress.pickle']))) and plot_progress:
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
    for epoch in range(start_epoch, epochs):
        train_loss.reset_states()
        val_loss.reset_states()

        # Get batches
        #x_batches, y_batches = get_batches(x, y, b_size=b_size, shuffle=True, seed=epoch)
        x_batches = x
        y_batches = y
        # Set-up training progress bar
        n_batch = len(x_batches)
        print("\nepoch {}/{}".format(epoch + 1, epochs))
        pb_i = Progbar(n_batch * b_size, stateful_metrics=['Loss: '])

        for (batch_num, (x_batch_i, y_batch_i)) in enumerate(zip(x_batches, y_batches)):
            x_batch = x_batches[x_batch_i]
            y_batch = y_batches[y_batch_i]

            x_batch = tf.constant(x_batch, dtype='float32')
            y_batch = tf.constant(y_batch, dtype='float32')

            train_step(inp=x_batch, tar=y_batch)

            # Print progbar
            if batch_num % summary_res == 0:
                values = [('Loss: ', train_loss.result())]
                pb_i.add(b_size*summary_res, values=values)

        # -------------------------------------------------------------------------------------------------------------
        # Validate the model
        # -------------------------------------------------------------------------------------------------------------

        # Get batches
        #x_batches, y_batches = get_batches(x_val, y_val, b_size=b_size, shuffle=True, seed=epoch)
        x_batches, y_batches = x_val, y_val
        for (batch_num, (x_batch_i, y_batch_i)) in enumerate(zip(x_batches, y_batches)):
            x_batch = x_batches[x_batch_i]
            y_batch = y_batches[y_batch_i]

            x_batch = tf.constant(x_batch, dtype='float32')
            y_batch = tf.constant(y_batch, dtype='float32')

            val_step(inp=x_batch, tar=y_batch)

        # Print validation losses:
        print('\nValidation Loss:', val_loss.result().numpy())

        # -------------------------
        # *** Checkpoint Model: ***
        # -------------------------
        if ckpt_flag:
            if (epoch % ckpt_interval == 0) or (val_loss.result() < min_val_error):
                to_save = tf.Module()
                to_save.inference = inference
                to_save.all_variables = list(transformer.variables)
                tf.saved_model.save(to_save, save_model_latest)

                if val_loss.result() < min_val_error:
                    print('*** New Best Model Saved to %s ***' % save_model_best)
                    to_save = tf.Module()
                    to_save.inference = inference
                    to_save.all_variables = list(transformer.variables)
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
                                              Validation=val_loss.result().numpy())

            else:
                fig_progress.on_epoch_end(Training=train_loss.result().numpy(),
                                          Validation=val_loss.result().numpy())

            # Store the plot if ckpting:
            if ckpt_flag:
                fig_progress.fig.savefig(os.path.normpath('/'.join([model_save_dir, save_folder, 'val_loss.png'])))
                if not os.path.exists(os.path.normpath('/'.join([model_save_dir, save_folder]))):
                    os.makedirs(os.path.normpath('/'.join([model_save_dir, save_folder])))
                pd.to_pickle(fig_progress, os.path.normpath('/'.join([model_save_dir, save_folder,
                                                                      'fig_progress.pickle'])))

        # Store currently best validation loss:
        if val_loss.result() < min_val_error:
            min_val_error = val_loss.result()

        # Append epoch losses to logs:
        _logs[0].append(train_loss.result().numpy())
        _logs[1].append(val_loss.result().numpy())


        if epoch == start_epoch:
            n_params = np.sum([np.prod(v.get_shape()) for v in transformer.variables])
            print('Number of parameters: ', n_params)

    # -----------------------------------------------------------------------------------------------------------------
    # Test the model
    # -----------------------------------------------------------------------------------------------------------------
    # # Load the best model:
    # if ckpt_flag:
    #     if os.path.exists(save_model_best):
    #         f = open('/'.join([os.path.dirname(save_model_best), 'epoch.txt']))
    #         ckpt_info = f.read()
    #         f.close()
    #         start_epoch = [int(s) for s in ckpt_info.split() if s.isdigit()][1]  # Get the latest epoch it trained
    #         print('Loading weights from best epoch ', start_epoch)
    #         loaded = tf.saved_model.load(save_model_best)
    #
    #         # Need to make a single prediction for the model as it needs to compile:
    #         transformer([tf.constant(x[:2], dtype='float32'),
    #                      tf.constant(y[:2][:, :-1, :], dtype='float32')],
    #                     training=False)
    #         for i in range(len(transformer.variables)):
    #             if transformer.variables[i].name != loaded.all_variables[i].name:
    #                 assert ValueError('Cannot load model, due to incompatible loaded and model...')
    #             transformer.variables[i].assign(loaded.all_variables[i].value())
    #
    # if 'n_params' not in locals():
    #     n_params = np.sum([np.prod(v.get_shape()) for v in transformer.variables])
    # if 'epoch' not in locals():
    #     _logs = [[0]]*len(_logs)
    #     epoch = 0
    #
    # # Get batches
    # x_batches, y_batches = get_batches(x_test, y_test, b_size=b_size, shuffle=False)
    # for (batch_num, (x_batch_i, y_batch_i)) in enumerate(zip(x_batches, y_batches)):
    #     x_batch = Z[x_batch_i]
    #     y_batch = Z[y_batch_i]
    #
    #     x_batch = tf.constant(x_batch, dtype='float32')
    #     y_batch = tf.constant(y_batch, dtype='float32')
    #
    #     val_step(inp=x_batch, tar=y_batch, testing=True)
    #
    # print('\n\nTest Loss: ', test_loss.result().numpy())
    # print('    Thres: ', test_loss_thres.result().numpy())
    # print('  SmThres: ', test_loss_sthres.result().numpy())
    # print('    MAE:   ', test_loss_mae.result().numpy(), '\n\n')
    #
    #
    # results = {
    #     'Test_Loss': test_loss.result().numpy(),
    #     'Test_Loss_Thres': test_loss_thres.result().numpy(),
    #     'Test_Loss_MAE': test_loss_mae.result().numpy(),
    #     'b_size': b_size,
    #     'num_layers': num_layers,
    #     'd_model': d_model,
    #     'dff': dff,
    #     'num_heads': num_heads,
    #     'drop': drop,
    #     'n_params': n_params,
    #     'learning_rate': learning_rate if isinstance(learning_rate, float) else 'Sched',
    #     'min_val_loss': np.min(_logs[1]),
    #     'min_train_loss': np.min(_logs[0]),
    #     'val_loss': _logs[1],
    #     'val_loss_thres': _logs[2],
    #     'val_loss_sthres': _logs[3],
    #     'val_loss_mae': _logs[4],
    #     'train_loss': _logs[0],
    # }

    # if ckpt_flag:
    #     with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
    #         for key, value in results.items():
    #             print('\n', key, '  : ', value, file=f)

    # -----------------------------------------------------------------------------------------------------------------
    # Save some Wav-file Predictions (from test set):
    # -----------------------------------------------------------------------------------------------------------------
    # TODO: Add filename instead of indices in saved name.. Also save losses.
    # TODO: Do this for the longformer as well...
    # if generate_wav is not None:
    #     np.random.seed(seed)
    #     gen_indxs = np.random.choice(len(y_test), generate_wav)
    #     x_gen = Z[x_test[gen_indxs]]
    #     y_gen = Z[y_test[gen_indxs]]
    #     predictions, _ = transformer([
    #         tf.constant(x_gen, dtype='float32'),
    #         tf.constant(y_gen[:, :-1, :], dtype='float32')],
    #         training=False)
    #     predictions = predictions.numpy()
    #     losses = [loss_fn(lab, pred).numpy() for (lab, pred) in zip(y_gen[:,1:,:], predictions)]
    #     predictions = scaler.inverse_transform(predictions)
    #     x_gen = scaler.inverse_transform(x_gen)
    #     y_gen = scaler.inverse_transform(y_gen)
    #     for i, indx in enumerate(gen_indxs):
    #         # Define directories
    #         pred_name = 'x' + str(x_test[gen_indxs][i]) + '_y' + str(y_test[gen_indxs][i]) + '_pred.wav'
    #         inp_name = 'x' + str(x_test[gen_indxs][i]) + '_y' + str(y_test[gen_indxs][i]) + '_inp.wav'
    #         tar_name = 'x' + str(x_test[gen_indxs][i]) + '_y' + str(y_test[gen_indxs][i]) + '_tar.wav'
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
    #         # Inverse STFT
    #         _, pred_i = signal.istft(predictions[i].T, nperseg=indeces['nperseg_i'], nfft=indeces['nfft'])
    #         _, inp_i = signal.istft(x_gen[i].T, nperseg=indeces['nperseg_i'], nfft=indeces['nfft'])
    #         _, tar_i = signal.istft(y_gen[i].T, nperseg=indeces['nperseg_i'], nfft=indeces['nfft'])
    #
    #         # Resample
    #         pred_i = signal.resample_poly(pred_i, up=44100 // indeces['samplerate'], down=1)
    #         pred_i = pred_i.astype('int16')
    #         inp_i = signal.resample_poly(inp_i, up=44100 // indeces['samplerate'], down=1)
    #         inp_i = inp_i.astype('int16')
    #         tar_i = signal.resample_poly(tar_i, up=44100 // indeces['samplerate'], down=1)
    #         tar_i = tar_i.astype('int16')
    #
    #         # Save Wav files
    #         wavfile.write(pred_dir, 44100, pred_i)
    #         wavfile.write(inp_dir, 44100, inp_i)
    #         wavfile.write(tar_dir, 44100, tar_i)
    #
    # return results

    # # TODO: Sort this out for proper training (i.e. not for a single sample as here... )
    # predictions, _ = transformer([x_batch, y_batch[:, :-1, :]], training=False)
    # #print(loss_fn(y_batch[:, 1:], predictions).numpy())
    #
    # # checking one prediction
    # #for i in range(len(x_batch[0,0,:])):
    #  #   predictions, _ = transformer([x_batch[:,:,:i+1], y_batch[:, -1, :i+1]], training=False)
    #   #  prediction[i] = predictions[0]
    #
    # predictions = predictions[0].numpy()
    # predictions = scaler.inverse_transform(predictions)
    # predictions = tf.squeeze(predictions).numpy()       # Remove dimensions of size == 1.
    #
    #
    # # To perform inverse STFT we need to have the same frequency bins as were produced by the original STFT (i.e.
    # # before removing higher frequencies), we therefore pad these values with zeroes:
    # #predictions = np.concatenate([predictions, np.zeros((predictions.shape[0], predictions.shape[-1]))], axis=-1)
    #
    # _, prediction = signal.istft(predictions.T, nperseg=4410, nfft=5120)     # Inverse STFT
    # prediction = signal.resample_poly(prediction, up=5, down=1)
    # prediction = prediction.astype('int16')   # Convert the data to int16 values.
    # wavfile.write(r'prediction.wav', 44100, prediction)


if __name__ == '__main__':
    data_dir = '/Users/riccardosimionato/Datasets/VA/VA_results'

    data = get_data(data_dir, seed=422)
    train_RAMT(
        data_dir=data_dir,
        data=data,
        model_save_dir='../../Transformer_TrainedModels',
        save_folder='Transformer_TESTING',
        ckpt_flag=True,
        plot_progress=True,
        b_size=16,
        num_layers=2,
        d_model=32,
        dff=32,
        num_heads=2,
        drop=0.1,
        epochs=55,
        seed=422,
        generate_wav=10)

