import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)
results = pd.read_pickle('../../HyperTuning/results_longformer.pickle')

# results['test_dist'] = results['test_dist_mean'] + results['test_dist_std']

# vars = ['b_size', 'num_layers', 'd_model', 'dff', 'num_heads', 'drop']
vars = ['b_size', 'num_layers', 'attention_window', 'd_model', 'dff', 'num_heads', 'drop', 'global_attention_prob']
# results = results.iloc[np.random.choice(np.arange(results.shape[0]), 200, replace=False),:]

# for CondSame in [True, False]:
#        fig, axs = plt.subplots(3, 6)
#        axs = axs.flatten()
#        for i, var in enumerate(vars):
#               axs[i].set_title(var)
#               for _, row in results[results['CondSame'] == CondSame].iterrows():
#                      axs[i].scatter(row[var], row['test_dist'], alpha=0.5, c='blue')

# hist_range = (np.min(results['test_dist']), np.max(results['test_dist']))
tar_var = 'min_val_loss'

# results = results[results[tar_var] < .005]
hist_range = (np.min(results[tar_var]), np.max(results[tar_var]))
number_of_bins = 60
# for CondSame in [True, False]:
       # sub_results = results[results['CondSame'] == CondSame]
sub_results = results
fig, axs = plt.subplots(2, 4)
axs = axs.flatten()
for i, var in enumerate(vars):
       data_sets = []
       # u = sub_results[var].unique()
       u = np.unique(sub_results[var])
       for u_i in u:
              # data_sets.append(sub_results['test_dist'][results[var] == u_i].to_numpy())
              data_sets.append(sub_results[tar_var][sub_results[var] == u_i].to_numpy())

       binned_data_sets = [np.histogram(d, range=hist_range, bins=number_of_bins)[0] for d in data_sets]
       binned_data_sets = [binned / np.sum(binned) for binned in binned_data_sets]
       binned_maximums = np.max(binned_data_sets, axis=1)
       # x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))
       x_locations = np.arange(0, len(binned_maximums)) * np.max(binned_maximums)

       bin_edges = np.linspace(hist_range[0], hist_range[1], number_of_bins + 1)
       centers = 0.5*(bin_edges + np.roll(bin_edges, 1))[:-1]
       heights = np.diff(bin_edges)

       for x_loc, binned_data in zip(x_locations, binned_data_sets):
              lefts = x_loc - 0.5*binned_data
              axs[i].barh(centers, binned_data, height=heights, left=lefts)

       axs[i].set_xticks(x_locations)
       try:
              axs[i].set_xticklabels(u)
       except:
              print('hello')
       axs[i].set_title(var)

plt.show()

plt.subplots()
plt.scatter(results['n_params'], results['min_val_loss'])
print('hello')

