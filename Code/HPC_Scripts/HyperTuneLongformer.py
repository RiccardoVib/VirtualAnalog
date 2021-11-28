from TrainRAMT_2 import train_RAMT, get_data
import numpy as np
import pandas as pd
import os
import pickle
import sys


data_dir='/scratch/users/larsbent/RAMT/Data/'
results_dir = '/scratch/users/larsbent/RAMT/HyperTuning/results_longformer.pickle'

print('Please Enter the seed to use: ')
# SEED = int(input())     # Get user input for seed
SEED = int(sys.argv[1])
print('Thanks. Seed entered: ', SEED, ' of type ', type(SEED))

for i in range(100):
    # if parameters are in file then choose a new one
    exist_in_results = True
    while exist_in_results:
        np.random.seed(SEED)

        epochs = 15
        b_size = np.random.choice([8, 16, 32], p=[0.3, 0.5, 0.2])
        num_layers = np.random.choice([2, 4, 8])
        attention_window = [np.random.choice([16, 32, 64, 128])]*num_layers
        d_model = np.random.choice([16, 32, 64, 128, 256])
        dff = np.random.choice([16, 32, 64, 128, 256])
        num_heads = np.random.choice([2, 4, 8])
        drop = np.random.choice([0., 0.2])
        glob_attention_prob = np.random.uniform(0, 0.07)

        if os.path.exists(results_dir):
            old_results = pd.read_pickle(results_dir)

            for _, row in old_results.iterrows():
                equals = [
                    b_size == row['b_size'],
                    num_layers == row['num_layers'],
                    attention_window == row['attention_window'],
                    d_model == row['d_model'],
                    dff == row['dff'],
                    num_heads == row['num_heads'],
                    drop == row['drop'],
                    glob_attention_prob == row['glob_attention_prob']]
                if all(equals):
                    break
                exist_in_results = False
        else:
            exist_in_results = False

    results = train_RAMT(
        data_dir=data_dir,
        ckpt_flag=False,
        plot_progress=False,
        b_size=b_size,
        num_layers=num_layers,
        attention_window=attention_window,
        d_model=d_model,
        dff=dff,
        num_heads=num_heads,
        drop=drop,
        glob_attention_prob=glob_attention_prob,
        epochs=epochs,
        seed=422)

    new_results = pd.concat({k: pd.Series([v]) for k, v in results.items()}, axis=1)

    if os.path.exists(results_dir):
        old_results = pd.read_pickle(results_dir)

        results_df = pd.concat([old_results, new_results], axis=0, ignore_index=True)
        del old_results
    else:
        results_df = new_results

    # pd.to_pickle(results_df, results_dir)
    pickle.dump(results_df, open(results_dir, 'wb'))

    del results_df, new_results, results

    SEED += 1