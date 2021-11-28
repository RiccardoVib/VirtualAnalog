from LSTM_2 import trainLSTM


# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
# print("tf version ", tf.__version__)
# print("keras version ", tf.keras.__version__)
#
# strategy = tf.distribute.MirroredStrategy()
# G = strategy.num_replicas_in_sync
# b_size = 16*G       # run half the batch on each GPU


data_dir = '/scratch/users/larsbent/RAMT/Data/',
seed = 422
trainLSTM(data_dir=data_dir,
          model_save_dir='/scratch/users/larsbent/RAMT/TrainedModels/',
          save_folder='LSTM_1',
          ckpt_flag=True,
          b_size=3,
          learning_rate=0.0001,
          encoder_units=[64, 64],
          decoder_units=[64, 64],
          dff_output=512,
          epochs=10,
          generate_wav=2)

trainLSTM(data_dir=data_dir,
          model_save_dir='/scratch/users/larsbent/RAMT/TrainedModels/',
          save_folder='LSTM_1',
          ckpt_flag=True,
          b_size=3,
          learning_rate=0.0001,
          encoder_units=[32, 32],
          decoder_units=[32, 32],
          dff_output=256,
          epochs=10,
          generate_wav=2)

trainLSTM(data_dir=data_dir,
          model_save_dir='/scratch/users/larsbent/RAMT/TrainedModels/',
          save_folder='LSTM_1',
          ckpt_flag=True,
          b_size=3,
          learning_rate=0.0001,
          encoder_units=[64, 64, 64],
          decoder_units=[64, 64, 64],
          dff_output=256,
          epochs=10,
          generate_wav=2)

trainLSTM(data_dir=data_dir,
          model_save_dir='/scratch/users/larsbent/RAMT/TrainedModels/',
          save_folder='LSTM_1',
          ckpt_flag=True,
          b_size=3,
          learning_rate=0.0001,
          encoder_units=[128],
          decoder_units=[128],
          dff_output=512,
          epochs=10,
          generate_wav=2)
