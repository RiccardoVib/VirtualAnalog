from Code.LSTM.LSTM import trainLSTM
from Code.LSTM.DenseFeed import trainDense
from Code.GetData import get_data

if __name__ == '__main__':
    #data_dir = '/Users/riccardosimionato/Datasets/VA/VA_results'
    data_dir = 'C:/Users/riccarsi/Documents/GitHub/VA_pickle'
    seed = 422
    data = get_data(data_dir=data_dir, seed=seed)
    trainDense(data_dir=data_dir,
              model_save_dir='../../../TrainedModels',
              save_folder='DenseFeed_Testing',
              ckpt_flag=True,
              b_size=16,
              learning_rate=0.0001,
              first_unit=[2, 2],
              epochs=1,
              data=data,
              generate_wav=2,
              shuffle_data=False)

    trainLSTM(data_dir=data_dir,
              model_save_dir='../../../TrainedModels',
              save_folder='LSTM_Testing',
              ckpt_flag=True,
              b_size=16,
              learning_rate=0.0001,
              encoder_units=[3, 2],
              decoder_units=[2, 2],
              epochs=1,
              data=data,
              generate_wav=2,
              shuffle_data=False)