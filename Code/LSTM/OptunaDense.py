import optuna
from Code.LSTM.DenseFeed import trainDense
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

N_TRIALS = 1000
#save_dir = r'C:/Users/riccarsi/Documents/GitHub/OptunaStudy/study_Dense.pkl'
#data_dir = 'C:/Users/riccarsi/Documents/GitHub/VA_pickle'
save_dir = r'../../Files/study_Dense.pkl'
data_dir = '../../Files'
epochs = 20
seed = 422


def visualise_study(data):
    df = data.trials_dataframe()
    df.dropna(inplace=True)
    df.reset_index(inplace=True)

    df['time'] = df.datetime_complete - df.datetime_start
    df['time'] = df.time.astype(np.int64) / (1_000_000_000)
    df = df[df.time > 0]

    # names = []
    #
    # for col in df.columns.values:
    #     if col[1] == '':
    #         names.append(col[0])
    #     else:
    #         names.append(col[1])
    #
    # df.columns = names

    print('best val:', round(df.value.min(), 4))
    a = sns.lineplot(x=df.index, y=df.value.cummin())
    a.set_xlabel('trial number')
    sns.scatterplot(x=df.index, y=df.value, color='red')
    a.set_ylabel('MinValLoss')
    a.legend(['best value', "trial's value"])


def objective(trial):
    # joblib.dump(study, save_dir)

    drop = trial.suggest_uniform('drop', 0, 0.4)
    learning_rate = trial.suggest_uniform('learning_rate', -5, -2)
    learning_rate = 10**learning_rate
    opt_type = trial.suggest_categorical('opt_type', ['SGD', 'Adam'])
    #loss_type = trial.suggest_categorical('loss_type', ['mae', 'mse'])
    units = trial.suggest_int('units', 2, 7)
    units = units**2
    layers = trial.suggest_int('layers', 1, 4)
    units = [units]*layers
    shuff = trial.suggest_categorical('shuffle_data', [False, True])


    val_loss = trainDense(
        data_dir=data_dir,
        epochs=epochs,
        seed=seed,
        ckpt_flag=False,
        #loss_type=loss_type,
        b_size=8,
        drop=drop,
        opt_type=opt_type,
        learning_rate=learning_rate,
        units=units,
        generate_wav=None,
        inference=False,
        shuffle_data=shuff
    )


    val_loss = val_loss['Min_val_loss']

    return val_loss


#if os.path.exists(save_dir):
    # study = joblib.load(save_dir)
#    study = optuna.load_study(study_name='lstm_tuning', storage='sqlite:///example_lstm.db')
#else:
study = optuna.create_study(direction='minimize', study_name="dense_tuning",
                            storage="sqlite:///example_dense.db", load_if_exists=True)

#study.add_trials(study.trials)
#joblib.dump(study, save_dir)
#print(wucewon)

study.optimize(objective, n_trials=N_TRIALS)

# visualise_study(study)

print('Best Parameters: ', study.best_params)
print('Best Val Loss: ', study.best_value)
print('Best Trial: ', study.best_trial)
print('All Trials: ', study.trials)
