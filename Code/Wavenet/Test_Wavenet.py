from TrainWavenet import trainWavenet

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# print("tf version ", tf.__version__)
# print("keras version ", tf.keras.__version__)
#
# strategy = tf.distribute.MirroredStrategy()
# G = strategy.num_replicas_in_sync
# b_size = 16*G       # run half the batch on each GPU


#data_dir = '/mnt/riccardo/'
data_dir = '/Users/riccardosimionato/Datasets/bach_multiinstrumental'
seed = 422

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


trainWavenet(data_dir=data_dir,
             model_save_dir='../../TrainedModels',
             save_folder='Wavenet_2',
             ckpt_flag=False,
             plot_progress=True,
             epochs=4000,
             generate_wav=10,
             dilation_rate=4,
             kernel_size=[1, 3, 1, 1],
             filters=[128, 128, 513],
             residual_channels=128,
             gate_channels=256,
             b_size=16,
             learning_rate=0.0001,
             inference_flag=True)