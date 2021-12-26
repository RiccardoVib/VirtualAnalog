
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

    if ckpt_flag:
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

    if ckpt_flag:
        with open(os.path.normpath('/'.join([model_save_dir, save_folder, 'results.txt'])), 'w') as f:
            for key, value in results.items():
                print('\n', key, '  : ', value, file=f)

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
