from Code.GetData import get_data
import os
import tensorflow as tf
import numpy as np
import pickle

from scipy.io import wavfile
from scipy import signal
from Code.Models import Transformer
from Code.TrainFunctionality import CustomSchedule, get_batches, PlotLossesSame, PlotLossesSubPlots
from tensorflow.keras.utils import Progbar
from Code.LossFunction import threshold_loss, threshold_loss_log
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
        x, y, x_val, y_val, x_test, y_test, Z, scaler, zero_value, indeces, name = get_data(data_dir=data_dir, seed=seed)
    else:
        x, y, x_val, y_val, x_test, y_test, Z, scaler, zero_value, indeces, name = data

    # x = np.array([[101], [100], [98]])
    # y = np.array([90, 97, 99])
    # x_val = x
    # y_val = y
    # x_test = x
    # y_test = y

    # -----------------------------------------------------------------------------------------------------------------
    # Set-up model, optimiser, lr_sched and losses:
    # -----------------------------------------------------------------------------------------------------------------
    model_save_dir = kwargs.get('model_save_dir', r'C:\Users\larsbent\Downloads')   # TODO: Change
    save_folder = kwargs.get('save_folder', 'Transformer')
    generate_wav = kwargs.get('generate_wav', None)
    ckpt_flag = kwargs.get('ckpt_flag', False)
    opt_type = kwargs.get('opt_type', 'Adam')
    plot_progress = kwargs.get('plot_progress', True)
    max_length = Z.shape[1]
    learning_rate = kwargs.get('learning_rate', None)
    b_size = kwargs.get('b_size', 16)
    num_layers = kwargs.get('num_layers', 4)
    d_model = kwargs.get('d_model', 128)
    dff = kwargs.get('dff', 512)
    num_heads = kwargs.get('num_heads', 8)
    drop = kwargs.get('drop', .2)
    output_dim = Z.shape[-1]
    inference_flag = kwargs.get('inference_flag', False)
    device_num = kwargs.get('device_num', None)
    loss_type = kwargs.get('loss_type', 'thres_log')

    # if device_num is not None:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = "device_num"

    if learning_rate is None:
        learning_rate = CustomSchedule(d_model=d_model, warmup_steps=4000)
    if opt_type == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    elif opt_type == 'SGD':
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    else:
        raise ValueError('Please pass opt_type as either Adam or SGD')

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

    # loss_fn = loss_fn()#tf.keras.losses.MeanAbsoluteError()

    if loss_type == 'thres_log':
        loss_fn = threshold_loss_log
    elif loss_type == 'thres':
        loss_fn = threshold_loss
    elif loss_type == 'mae':
        loss_fn = tf.keras.losses.MeanAbsoluteError()
    elif loss_type == 'mse':
        loss_fn = tf.keras.losses.MeanSquaredError()
    else:
        raise ValueError('Please give a valid loss function type')

    mae_fn = tf.keras.losses.MeanAbsoluteError()       # loss_fn(sampling_rate=8820, a=.5, b=.5)
    mse_fn = tf.keras.losses.MeanSquaredError()
    # loss_fn = tf.keras.losses.MeanSquaredError()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_loss_mae = tf.keras.metrics.Mean(name='val_loss_mae')
    val_loss_mse = tf.keras.metrics.Mean(name='val_loss_mse')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_loss_mae = tf.keras.metrics.Mean(name='test_loss_mae')
    test_loss_mse = tf.keras.metrics.Mean(name='test_loss_mse')

    # -----------------------------------------------------------------------------------------------------------------
    # Define the training functionality
    # -----------------------------------------------------------------------------------------------------------------
    @tf.function
    def train_step(inp, tar):
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]

        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp], training=True)

            # loss = loss_thres = loss_mae = loss_fn(tar_real, predictions)

            loss = loss_fn(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        opt.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss.update_state(loss)

    @tf.function
    def val_step(inp, tar, testing=False):
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]

        predictions, attn_weights = transformer([inp, tar_inp], training=False)

        # loss = loss_thres = loss_mae = loss_fn(tar_real, predictions)
        loss = loss_fn(tar_real, predictions)
        loss_mae = mae_fn(tar_real, predictions)
        loss_mse = mse_fn(tar_real, predictions)

        if not testing:
            val_loss.update_state(loss)
            val_loss_mae.update_state(loss_mae)
            val_loss_mse.update_state(loss_mse)
        else:
            test_loss.update_state(loss)
            test_loss_mae.update_state(loss_mae)
            test_loss_mse.update_state(loss_mse)

        return attn_weights

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
            if inference_flag:
                loaded = tf.saved_model.load(save_model_best)
            else:
                loaded = tf.saved_model.load(save_model_latest)

            # Need to make a single prediction for the model as it needs to compile:
            transformer([tf.constant(Z[x[:2]], dtype='float32'),
                         tf.constant(Z[y[:2]][:, :-1, :], dtype='float32')],
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
    _logs = [[], [], [], []]
    min_val_error = np.inf
    summary_res = 1
    if inference_flag:
        start_epoch = epochs
    for epoch in range(start_epoch, epochs):
        train_loss.reset_states()
        val_loss.reset_states()
        val_loss_mae.reset_states()
        val_loss_mse.reset_states()

        # Get batches
        x_batches, y_batches = get_batches(x, y, b_size=b_size, shuffle=True, seed=epoch)

        # Set-up training progress bar
        n_batch = len(x_batches)
        print("\nepoch {}/{}".format(epoch + 1, epochs))
        pb_i = Progbar(n_batch * b_size, stateful_metrics=['Loss: '])

        for (batch_num, (x_batch_i, y_batch_i)) in enumerate(zip(x_batches, y_batches)):
            x_batch = Z[x_batch_i]
            y_batch = Z[y_batch_i]

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
        x_batches, y_batches = get_batches(x_val, y_val, b_size=b_size, shuffle=True, seed=epoch)

        for (batch_num, (x_batch_i, y_batch_i)) in enumerate(zip(x_batches, y_batches)):
            x_batch = Z[x_batch_i]
            y_batch = Z[y_batch_i]

            x_batch = tf.constant(x_batch, dtype='float32')
            y_batch = tf.constant(y_batch, dtype='float32')

            val_step(inp=x_batch, tar=y_batch)

        # Print validation losses:
        print('\nValidation Loss:', val_loss.result().numpy())
        print('         MAE:   ', val_loss_mae.result().numpy())
        print('         MSE:   ', val_loss_mse.result().numpy())

        # -------------------------
        # *** Checkpoint Model: ***
        # -------------------------
        if ckpt_flag:
            if (epoch % ckpt_interval == 0) or (val_loss_mse.result() < min_val_error):
                to_save = tf.Module()
                to_save.inference = inference
                to_save.all_variables = list(transformer.variables)
                tf.saved_model.save(to_save, save_model_latest)

                if val_loss_mse.result() < min_val_error:
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
                # fig_progress = PlotLossesSame(epoch + 1,
                #                               Training=train_loss.result().numpy(),
                #                               Validation=val_loss.result().numpy(),
                #                               Val_Thres=val_loss_thres.result().numpy(),
                #                               Val_sThres=val_loss_sthres.result().numpy(),
                #                               Val_MAE=val_loss_mae.result().numpy(),
                #                               Val_MSE=val_loss_mse.result().numpy())
                fig_progress = PlotLossesSubPlots(epoch + 1,
                                                  Losses1={
                                                      'Training': train_loss.result().numpy(),
                                                      'Validation': val_loss.result().numpy(),
                                                  },
                                                  Losses2={
                                                      'Val_MAE': val_loss_mae.result().numpy(),
                                                      'Val_MSE': val_loss_mse.result().numpy()
                                                  })
            else:
                # fig_progress.on_epoch_end(Training=train_loss.result().numpy(),
                #                           Validation=val_loss.result().numpy(),
                #                           Val_Thres=val_loss_thres.result().numpy(),
                #                           Val_sThres=val_loss_sthres.result().numpy(),
                #                           Val_MAE=val_loss_mae.result().numpy(),
                #                           Val_MSE=val_loss_mse.result().numpy())
                fig_progress.on_epoch_end(
                    Losses1={
                        'Training': train_loss.result().numpy(),
                        'Validation': val_loss.result().numpy(),
                    },
                    Losses2={
                        'Val_MAE': val_loss_mae.result().numpy(),
                        'Val_MSE': val_loss_mse.result().numpy()
                    })

            # Store the plot if ckpting:
            if ckpt_flag:
                fig_progress.fig.savefig(os.path.normpath('/'.join([model_save_dir, save_folder, 'val_loss.png'])))
                if not os.path.exists(os.path.normpath('/'.join([model_save_dir, save_folder]))):
                    os.makedirs(os.path.normpath('/'.join([model_save_dir, save_folder])))
                pd.to_pickle(fig_progress, os.path.normpath('/'.join([model_save_dir, save_folder,
                                                                      'fig_progress.pickle'])))

        # Store currently best validation loss:
        if val_loss_mse.result() < min_val_error:
            min_val_error = val_loss_mse.result()

        # Append epoch losses to logs:
        _logs[0].append(train_loss.result().numpy())
        _logs[1].append(val_loss.result().numpy())
        _logs[2].append(val_loss_mae.result().numpy())
        _logs[3].append(val_loss_mse.result().numpy())

        if epoch == start_epoch:
            n_params = np.sum([np.prod(v.get_shape()) for v in transformer.variables])
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
            transformer([tf.constant(Z[x[:2]], dtype='float32'),
                         tf.constant(Z[y[:2]][:, :-1, :], dtype='float32')],
                        training=False)
            for i in range(len(transformer.variables)):
                if transformer.variables[i].name != loaded.all_variables[i].name:
                    assert ValueError('Cannot load model, due to incompatible loaded and model...')
                transformer.variables[i].assign(loaded.all_variables[i].value())

    if 'n_params' not in locals():
        n_params = np.sum([np.prod(v.get_shape()) for v in transformer.variables])
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
    print('    MAE:   ', test_loss_mae.result().numpy())
    print('    MSE:   ', test_loss_mse.result().numpy(), '\n\n')

    results = {
        'Test_Loss': test_loss.result().numpy(),
        'Test_Loss_MAE': test_loss_mae.result().numpy(),
        'Test_Loss_MSE': test_loss_mse.result().numpy(),
        'b_size': b_size,
        'loss_type': loss_type,
        'num_layers': num_layers,
        'd_model': d_model,
        'dff': dff,
        'num_heads': num_heads,
        'drop': drop,
        'n_params': n_params,
        'learning_rate': learning_rate if isinstance(learning_rate, float) else 'Sched',
        'min_val_loss': np.min(_logs[1]),
        'min_val_MAE': np.min(_logs[2]),
        'min_val_MSE': np.min(_logs[3]),
        'min_train_loss': np.min(_logs[0]),
        'val_loss': _logs[1],
        'val_loss_mae': _logs[2],
        'val_loss_mse': _logs[3],
        'train_loss': _logs[0],
    }

    if ckpt_flag and not inference_flag:
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)
            pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

    # -----------------------------------------------------------------------------------------------------------------
    # Save some Wav-file Predictions (from test set):
    # -----------------------------------------------------------------------------------------------------------------
    if generate_wav is not None:
        np.random.seed(333)
        gen_indxs = np.random.choice(len(y_test), generate_wav)
        x_gen = Z[x_test[gen_indxs]]
        y_gen = Z[y_test[gen_indxs]]
        predictions, _ = transformer([
            tf.constant(x_gen, dtype='float32'),
            tf.constant(y_gen[:, :-1, :], dtype='float32')],
            training=False)
        predictions = predictions.numpy()
        losses = [np.sum(threshold_loss(tf.cast(lab, tf.float32), tf.cast(pred, tf.float32))) for (lab, pred) in zip(y_gen[:,1:,:], predictions)]
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
            plot_spectral(Zxx=predictions[i], title='Predictions_' + str(np.round(losses[i], 5)),
                          save_dir=os.path.normpath(os.path.join(spectral_dir, pred_name)).replace('.wav', '.png'))
            plot_spectral(Zxx=x_gen[i], title='Inputs',
                          save_dir=os.path.normpath(os.path.join(spectral_dir, inp_name)).replace('.wav', '.png'))
            plot_spectral(Zxx=y_gen[i], title='Target',
                          save_dir=os.path.normpath(os.path.join(spectral_dir, tar_name)).replace('.wav', '.png'))

            # Inverse STFT
            _, pred_i = signal.istft(predictions[i].T, nperseg=indeces[1]['nperseg_i'], nfft=indeces[1]['nfft'])
            _, inp_i = signal.istft(x_gen[i].T, nperseg=indeces[1]['nperseg_i'], nfft=indeces[1]['nfft'])
            _, tar_i = signal.istft(y_gen[i].T, nperseg=indeces[1]['nperseg_i'], nfft=indeces[1]['nfft'])

            # Resample
            pred_i = signal.resample_poly(pred_i, up=44100 // indeces[1]['samplerate'], down=1)
            pred_i = pred_i.astype('int16')
            inp_i = signal.resample_poly(inp_i, up=44100 // indeces[1]['samplerate'], down=1)
            inp_i = inp_i.astype('int16')
            tar_i = signal.resample_poly(tar_i, up=44100 // indeces[1]['samplerate'], down=1)
            tar_i = tar_i.astype('int16')

            # Save Wav files
            wavfile.write(pred_dir, 44100, pred_i)
            wavfile.write(inp_dir, 44100, inp_i)
            wavfile.write(tar_dir, 44100, tar_i)

    return results

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


def train_Longformer(data_dir, epochs, seed=422, data=None, **kwargs):
    # Get the data:
    if data is None:
        x, y, x_val, y_val, x_test, y_test, Z, scaler, zero_value, indeces, name = get_data(data_dir=data_dir, seed=seed)
    else:
        x, y, x_val, y_val, x_test, y_test, Z, scaler, zero_value, indeces, name = data
    # x_val = x
    # y_val = y
    # x_test = x
    # y_test = y
    # -----------------------------------------------------------------------------------------------------------------
    # Set-up model, optimiser, lr_sched and losses:
    # -----------------------------------------------------------------------------------------------------------------
    model_save_dir = kwargs.get('model_save_dir', r'C:\Users\larsbent\Downloads')     # TODO: Change
    save_folder = kwargs.get('save_folder', 'Longformer')
    ckpt_flag = kwargs.get('ckpt_flag', False)
    plot_progress = kwargs.get('plot_progress', True)
    b_size = kwargs.get('b_size', 16)
    max_length = Z.shape[1]
    num_layers = kwargs.get('num_layers', 3)
    d_model = kwargs.get('d_model', 128)
    learning_rate = kwargs.get('learning_rate', None)
    attention_window = kwargs.get('attention_window', [64, 64, 64])
    dff = kwargs.get('dff', 512)
    num_heads = kwargs.get('num_heads', 8)
    drop = kwargs.get('drop', .2)
    global_attention_prob = kwargs.get('global_attention_prob', 0)
    generate_wav = kwargs.get('generate_wav', None)
    inference_flag = kwargs.get('inference_flag', False)
    output_dim = Z.shape[-1]

    if learning_rate is None:
        learning_rate = CustomSchedule(d_model=d_model, warmup_steps=4000)
    opt = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    configuration = Longformer.LongformerConfig(attention_window=attention_window,
                                                pad_token_id=1,  # Pad token id (don't change)
                                                hidden_size=d_model,
                                                num_hidden_layers=num_layers,
                                                num_attention_heads=num_heads,
                                                intermediate_size=dff,  # For intermediate dense layer
                                                hidden_act="tanh",
                                                hidden_dropout_prob=drop,
                                                attention_probs_dropout_prob=drop,
                                                max_position_embeddings=max_length,
                                                initializer_range=0.02,
                                                layer_norm_eps=1e-12)
    longformer = Longformer.My_TFLongformerModel(configuration, output_dim=output_dim, act_fn_output='sigmoid')
    if global_attention_prob > 0:
        global_attention_mask = np.random.choice([0, 1], (1, tf.shape(Z)[1]),
                                                 p=[1 - global_attention_prob, global_attention_prob])
        glob_indices = list(np.argwhere(global_attention_mask == 1)[:, -1])
        global_attention_mask = tf.constant(global_attention_mask, 'int32')
        global_attention_mask = tf.repeat(global_attention_mask, b_size, axis=0)
    else:
        glob_indices = [99999]

    loss_fn = tf.keras.losses.MeanAbsoluteError()   # TODO: Change

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
    # @tf.function
    def train_step(inp, tar, global_attention_mask=None):
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]

        with tf.GradientTape() as tape:
            predictions = longformer(inputs_embeds=inp,
                                     inputs_embeds_dec=tar_inp,
                                     global_attention_mask=global_attention_mask,
                                     global_attention_mask_dec=global_attention_mask,
                                     output_attentions=False,
                                     output_hidden_states=False,
                                     training=True)
            predictions = predictions['output']
            loss_thres, loss_sthres = threshold_loss(tar_real, predictions, zero_value=0.)
            loss = loss_thres + loss_sthres
        gradients = tape.gradient(loss, longformer.trainable_variables)
        opt.apply_gradients(zip(gradients, longformer.trainable_variables))
        train_loss.update_state(loss)

    def val_step(inp, tar, testing=False, global_attention_mask=None):
        tar_inp = tar[:, :-1, :]
        tar_real = tar[:, 1:, :]

        outputs = longformer(inputs_embeds=inp,
                             inputs_embeds_dec=tar_inp,
                             global_attention_mask=global_attention_mask,
                             global_attention_mask_dec=global_attention_mask,
                             output_attentions=False,
                             output_hidden_states=False,
                             training=False)

        loss_thres, loss_sthres = threshold_loss(tar_real, outputs['output'], zero_value=0.)
        loss = loss_thres + loss_sthres
        loss_mae = loss_fn(tar_real, outputs['output'])

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

        return outputs

    # -----------------------------------------------------------------------------------------------------------------
    # Set up checkpointing (saving) of the model
    # -----------------------------------------------------------------------------------------------------------------
    start_epoch = 0
    if ckpt_flag:
        save_model_latest = os.path.normpath('/'.join([model_save_dir, save_folder, 'Latest', 'vars.pickle']))
        save_model_best = os.path.normpath('/'.join([model_save_dir, save_folder, 'Best', 'vars.pickle']))

        # Load weights if exists
        if os.path.exists(save_model_latest):
            f = open('/'.join([os.path.dirname(os.path.dirname(save_model_best)), 'epoch.txt']))
            ckpt_info = f.read()
            f.close()
            start_epoch = [int(s) for s in ckpt_info.split() if s.isdigit()][0]  # Get the latest epoch it trained
            print('Loading weights and starting from epoch ', start_epoch)
            if inference_flag:
                loaded_vars = pickle.load(open(save_model_best, 'rb'))
            else:
                loaded_vars = pickle.load(open(save_model_latest, 'rb'))

            # Need to make a single prediction for the model as it needs to compile:
            longformer(inputs_embeds=tf.constant(Z[x[:2]], dtype='float32'),
                       inputs_embeds_dec=tf.constant(Z[y[:2]][:,:-1,:], dtype='float32'),
                       global_attention_mask=global_attention_mask[:2,:],
                       global_attention_mask_dec=global_attention_mask[:2,:],
                       output_attentions=False,
                       output_hidden_states=False,
                       training=False)
            for i in range(len(longformer.variables)):
                if longformer.variables[i].name != loaded_vars[i].name:
                    assert ValueError('Cannot load model, due to incompatible loaded and model...')
                longformer.variables[i].assign(loaded_vars[i].value())
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
    _logs = [[]]*5
    min_val_error = np.inf
    summary_res = 2
    if inference_flag:
        start_epoch = epochs
    for epoch in range(start_epoch, epochs):
        train_loss.reset_states()
        val_loss.reset_states()
        val_loss_thres.reset_states()
        val_loss_mae.reset_states()
        val_loss_sthres.reset_states()

        # -----------------
        # *** Training: ***
        # -----------------
        x_batches, y_batches = get_batches(x, y, b_size=b_size, shuffle=True, seed=epoch)
        # Set-up training progress bar
        n_batch = len(x_batches)
        print("\nepoch {}/{}".format(epoch + 1, epochs))
        pb_i = Progbar(n_batch * b_size, stateful_metrics=['Loss: '])

        for batch_num, (x_batch_i, y_batch_i) in enumerate(zip(x_batches, y_batches)):
            # x_batch, y_batch = [], []
            # for i in range(len(x_batch_i)):
            #     x_batch.append(Z[x_batch_i[i][0]])
            #     y_batch.append(Z[y_batch_i[i]])
            # x_batch = tf.constant(x_batch, dtype='float32')
            # y_batch = tf.constant(y_batch, dtype='float32')
            x_batch = tf.constant(Z[x_batch_i], dtype='float32')
            y_batch = tf.constant(Z[y_batch_i], dtype='float32')
            train_step(inp=x_batch, tar=y_batch,
                       global_attention_mask_enc=global_attention_mask[:tf.shape(x_batch)[0],:])
            # Print progbar
            if batch_num % summary_res == 0:
                values = [('Loss: ', train_loss.result())]
                pb_i.add(b_size*summary_res, values=values)

        # -------------------
        # *** Validation: ***
        # -------------------
        x_batches, y_batches = get_batches(x_val, y_val, b_size=b_size, shuffle=True, seed=epoch)
        for batch_num, (x_batch_i, y_batch_i) in enumerate(zip(x_batches, y_batches)):
            x_batch = tf.constant(Z[x_batch_i], dtype='float32')
            y_batch = tf.constant(Z[y_batch_i], dtype='float32')
            val_step(inp=x_batch, tar=y_batch, global_attention_mask=global_attention_mask[:tf.shape(x_batch)[0],:])

        # Print validation losses:
        print('\nValidation Loss:    ', val_loss.result().numpy())
        print('         Thres:     ', val_loss_thres.result().numpy())
        print('         SmalThres: ', val_loss_sthres.result().numpy())
        print('         MAE:       ', val_loss_mae.result().numpy())

        # -------------------
        # *** Checkpoint Model: ***
        # -------------------
        if ckpt_flag:
            if (epoch % ckpt_interval == 0) or (val_loss.result() < min_val_error):
                if not os.path.exists(os.path.dirname(save_model_latest)):
                    os.makedirs(os.path.dirname(save_model_latest))
                    os.makedirs(os.path.dirname(save_model_best))

                pickle.dump(longformer.variables, open(save_model_latest, 'wb'))
                if val_loss.result() < min_val_error:
                    pickle.dump(longformer.variables, open(save_model_best, 'wb'))
                    best_epoch = epoch

                epoch_dir = '/'.join([os.path.dirname(os.path.dirname(save_model_best)), 'epoch.txt'])
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
                pd.to_pickle(fig_progress, os.path.normpath('/'.join([model_save_dir, save_folder,
                                                                      'fig_progress.pickle'])))

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
            n_params = np.sum([np.prod(v.get_shape()) for v in longformer.variables])
            print('Number of parameters: ', n_params)

    # -----------------------------------------------------------------------------------------------------------------
    # Test the model
    # -----------------------------------------------------------------------------------------------------------------
    if 'n_params' not in locals():
        n_params = np.sum([np.prod(v.get_shape()) for v in longformer.variables])
    if 'epoch' not in locals():
        _logs = [[0]]*len(_logs)
        epoch = 0

    if ckpt_flag and not inference_flag:
        loaded_vars = pickle.load(open(save_model_best, 'rb'))

        for i in range(len(longformer.variables)):
            if longformer.variables[i].name != loaded_vars[i].name:
                assert ValueError('Cannot load model, due to incompatible loaded and model...')
            longformer.variables[i].assign(loaded_vars[i].value())

    # Get batches
    x_batches, y_batches = get_batches(x_test, y_test, b_size=b_size, shuffle=False, seed=epoch)
    for (batch_num, (x_batch_i, y_batch_i)) in enumerate(zip(x_batches, y_batches)):
        x_batch, y_batch = [], []
        for i in range(len(x_batch_i)):
            x_batch.append(Z[x_batch_i[i]])
            y_batch.append(Z[y_batch_i[i]])

        x_batch = tf.constant(x_batch, dtype='float32')
        y_batch = tf.constant(y_batch, dtype='float32')

        val_step(inp=x_batch, tar=y_batch, global_attention_mask=global_attention_mask[:tf.shape(x_batch)[0], :],
                 testing=True)

    print('\n\n Test Loss: ', test_loss.result().numpy(), '\n\n')

    results = {
        'Test_Loss': test_loss.result().numpy(),
        'b_size': b_size,
        'num_layers': num_layers,
        'attention_window': attention_window,
        'd_model': d_model,
        'dff': dff,
        'num_heads': num_heads,
        'drop': drop,
        'n_params': n_params,
        'global_attention_prob': global_attention_prob,
        'learning_rate': learning_rate if isinstance(learning_rate, float) else 'Sched',
        'min_val_loss': np.min(_logs[1]),
        'min_train_loss': np.min(_logs[0]),
        'val_loss': _logs[1],
        'val_loss_thres': _logs[2],
        'val_loss_sthres': _logs[3],
        'val_loss_mae': _logs[4],
        'train_loss': _logs[0],
        'glob_indices': glob_indices,
    }

    if ckpt_flag and not inference_flag:
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)
        pickle.dump(results, open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.pkl'])), 'wb'))

    # -----------------------------------------------------------------------------------------------------------------
    # Save some Wav-file Predictions (from test set):
    # -----------------------------------------------------------------------------------------------------------------
    # TODO: Add filename instead of indices in saved name.. Also save losses.
    # TODO: Do this for the longformer as well...
    if generate_wav is not None:
        np.random.seed(seed)
        gen_indxs = np.random.choice(len(y_test), generate_wav)
        x_gen = Z[x_test[gen_indxs]]
        y_gen = Z[y_test[gen_indxs]]
        predictions = longformer(inputs_embeds=tf.constant(x_gen, dtype='float32'),
                                 inputs_embeds_dec=tf.constant(y_gen[:, :-1, :], dtype='float32'),
                                 global_attention_mask=global_attention_mask[:tf.shape(x_gen)[0], :],
                                 global_attention_mask_dec=global_attention_mask[:tf.shape(x_gen)[0], :],
                                 output_attentions=False,
                                 output_hidden_states=False,
                                 training=False)
        predictions = predictions['output']
        predictions = predictions.numpy()
        losses = [loss_fn(lab, pred).numpy() for (lab, pred) in zip(y_gen[:,1:,:], predictions)]
        predictions = scaler.inverse_transform(predictions)
        x_gen = scaler.inverse_transform(x_gen)
        y_gen = scaler.inverse_transform(y_gen)

        # predictions = LogscaleTransform.db_to_lin(predictions)
        # x_gen = LogscaleTransform.db_to_lin(x_gen)
        # y_gen = LogscaleTransform.db_to_lin(y_gen)


        for i, indx in enumerate(gen_indxs):
            # Define directories
            pred_name = 'x' + str(x_test[gen_indxs][i][0]) + '_y' + str(y_test[gen_indxs][i]) + '_pred.wav'
            inp_name = 'x' + str(x_test[gen_indxs][i][0]) + '_y' + str(y_test[gen_indxs][i]) + '_inp.wav'
            tar_name = 'x' + str(x_test[gen_indxs][i][0]) + '_y' + str(y_test[gen_indxs][i]) + '_tar.wav'

            pred_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', pred_name))
            inp_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', inp_name))
            tar_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'WavPredictions', tar_name))

            if not os.path.exists(os.path.dirname(pred_dir)):
                os.makedirs(os.path.dirname(pred_dir))

            # Save some Spectral Plots:
            spectral_dir = os.path.normpath(os.path.join(model_save_dir, save_folder, 'SpectralPlots'))
            if not os.path.exists(spectral_dir):
                os.makedirs(spectral_dir)
            plot_spectral(Zxx=predictions[i], title='Predictions' + str(np.round(losses[i], 5)),
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

    # predictions = longformer(inputs_embeds=x_batch[:1, :, :],
    #                          inputs_embeds_dec=y_batch[:1, :-1, :],
    #                          global_attention_mask=global_attention_mask[:1, :],
    #                          global_attention_mask_dec=global_attention_mask[:1, :],
    #                          output_attentions=False,
    #                          output_hidden_states=False,
    #                          training=False)
    # predictions = predictions['output']
    # print(loss_fn(y_batch[:, 1:], predictions).numpy())
    # predictions = predictions.numpy()
    # predictions = scaler.inverse_transform(predictions)
    # predictions = tf.squeeze(predictions).numpy()  # Remove dimensions of size == 1.
    #
    # # To perform inverse STFT we need to have the same frequency bins as were produced by the original STFT (i.e.
    # # before removing higher frequencies), we therefore pad these values with zeroes:
    # _, predictions = signal.istft(predictions.T, fs=44100, nperseg=4410)  # Inverse STFT
    # predictions = predictions.astype('int16')  # Convert the data to int16 values.
    # wavfile.write(r'C:\Users\larsbent\Downloads\predictions.wav', 44100, predictions)


if __name__ == '__main__':
    #data_dir = '../../Data/bach_multiinstrumental/bach_multiinstrumental/Training/pickles'
    data_dir = '/Users/riccardosimionato/Datasets/bach_multiinstrumental/CompleteDatasetsRAMT_'

    train_RAMT(
        data_dir=data_dir,
        model_save_dir=r'C:\Users\larsbent\Downloads\Temp', # '../../HyperTuning',
        save_folder='Transformer_TestingLoss',
        ckpt_flag=True,
        plot_progress=True,
        loss_type='thres_log',
        b_size=16,
        learning_rate=0.0001,
        num_layers=3,
        d_model=64,
        dff=128,
        num_heads=2,
        drop=0.2,
        epochs=20,
        seed=422,
        generate_wav=10,
        inference_flag=False)

    # train_Longformer(
    #     data_dir=data_dir,
    #     data=data,
    #     model_save_dir='../../HyperTuning',
    #     save_folder='Longformer_1',
    #     ckpt_flag=False,
    #     plot_progress=True,
    #     learning_rate=0.0001,
    #     b_size=16,
    #     num_layers=2,
    #     d_model=32,
    #     attention_window=[32, 32, 32, 32],
    #     dff=128,
    #     num_heads=4,
    #     drop=0.2,
    #     global_attention_prob=.005,
    #     epochs=1486,
    #     seed=422,
    #     generate_wav=10,
    #     inference_flag=True)