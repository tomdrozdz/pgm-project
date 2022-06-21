import os
import sys

import pandas as pd

from src.lstm import get_param_results
import pickle
import matplotlib.pyplot as plt

OUT_TO_FILE = True
RES_FILE_1 = 'hp_tuning_1.pkl'
RES_FILE_2 = 'hp_tuning_2.pkl'

with open(RES_FILE_1, 'rb') as f:
    res_df_1 = pickle.load(file=f)
regression = True
res_hidden = get_param_results(res_df_1, 'hidden_dim', regression)
res_layers = get_param_results(res_df_1, 'no_layers', regression)
res_seq_len = get_param_results(res_df_1, 'sequence_length', regression)

fig, ax = plt.subplots(2, 3, figsize=(10, 8))


def param_chart(ax, res_df, param_name, y_label, x_label, metric='mse'):
    mean_key = metric+'_mean'
    std_key = metric+'_std'
    ax.bar(range(len(res_df[mean_key])), res_df[mean_key])
    ax.errorbar(range(len(res_df[mean_key])), res_df[mean_key],
                yerr=res_df[std_key], color='black', ls='none', capsize=5)
    ax.set_xticks(range(len(res_df[mean_key])))
    ax.set_xticklabels(res_df[param_name])
    if y_label is not None:
        ax.set_ylabel(x_label)
    ax.set_xlabel(y_label)


param_chart(ax[0][0], res_hidden, 'hidden_dim', 'LSTM hidden dimension', 'MSE Value')
param_chart(ax[0][1], res_layers, 'no_layers', 'LSTM number of layers', None)
param_chart(ax[0][2], res_seq_len, 'sequence_length', 'Sequence length', None)

param_chart(ax[1][0], res_hidden, 'hidden_dim', 'LSTM hidden dimension', 'F1 (macro) Value', metric='f1')
param_chart(ax[1][1], res_layers, 'no_layers', 'LSTM number of layers', None, metric='f1')
param_chart(ax[1][2], res_seq_len, 'sequence_length', 'Sequence length', None, metric='f1')


fig.suptitle('Hyperparameters tuning results #1')
fig.tight_layout()
fig.savefig('results/lstm/hp_results_1.png')

best_params_1 = res_df_1[res_df_1['regression'] == regression].sort_values(['mse_mean']).iloc[0,:].to_dict()

with open(RES_FILE_2, 'rb') as f:
    res_df_2 = pickle.load(file=f)

regression = True
res_drop = get_param_results(res_df_2, 'drop_prob', regression, other_params=best_params_1)
res_le = get_param_results(res_df_2, 'lr', regression, other_params=best_params_1)

fig, ax = plt.subplots(2, 2, figsize=(6, 6))
param_chart(ax[0][0], res_drop, 'drop_prob', 'Dropout probability', 'MSE Value')
param_chart(ax[0][1], res_le, 'lr', 'Learning rate', None)
param_chart(ax[1][0], res_drop, 'drop_prob', 'Dropout probability', 'F1 (macro) Value', metric='f1')
param_chart(ax[1][1], res_le, 'lr', 'Learning rate', None, metric='f1')

fig.suptitle('Hyperparameters tuning results #2')
fig.tight_layout()
fig.savefig('results/lstm/hp_results_2.png')

best_params_2 = res_df_2[res_df_2['regression'] == True].sort_values(['mse_mean']).iloc[0,:].to_dict()

if OUT_TO_FILE:
    sys.stdout = open('results/lstm/hp_tuning.txt', "w")
print('Example Learning curve for best regression model: ' + best_params_1['prefix_3']+'_mse.png')
print('\nBest parameters for regression model:')
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
columns = ['hidden_dim', 'no_layers', 'sequence_length', 'drop_prob', 'lr', 'mse_mean', 'f1_mean']
print(pd.DataFrame(best_params_2, index=[0])[columns])
best_params_2 = res_df_2[res_df_2['regression'] == False].sort_values(['mse_mean']).iloc[0,:].to_dict()
print('\nBest parameters for classification model:')
print(pd.DataFrame(best_params_2, index=[0])[columns])

