from TrainRAMT_2 import train_RAMT, train_Longformer


# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# print("tf version ", tf.__version__)
# print("keras version ", tf.keras.__version__)
#
# strategy = tf.distribute.MirroredStrategy()
# G = strategy.num_replicas_in_sync
# b_size = 16*G       # run half the batch on each GPU

train_Longformer(
    data_dir='/scratch/users/larsbent/RAMT/Data/',
    model_save_dir='/scratch/users/larsbent/RAMT/TrainedModels/',
    save_folder='Longformer_1',
    ckpt_flag=True,
    plot_progress=True,
    b_size=16,
    num_layers=2,
    attention_window=[128, 128],
    d_model=32,
    dff=256,
    num_heads=8,
    drop=.2,
    global_attention_prob=0.002,
    epochs=30,
    seed=422)
