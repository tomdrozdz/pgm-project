import pickle

import numpy as np
import pandas as pd
import torch

from src.data import load_data
from src.lstm import hp_tuning

FIRST_TUNING_FILE = 'hp_tuning_1.pkl'
SECOND_TUNING_FILE = 'hp_tuning_2.pkl'

MAX_EPOCHS = 300

torch.manual_seed(11)
np.random.seed(11)

device = 'cuda:0'

df = load_data()

param_grid_1 = {'sequence_length': [5, 10, 15], 'no_layers': [2, 3, 4], 'hidden_dim': [16, 32, 64], 'regression': [True, False],
                'drop_prob': [0.5], 'lr': [0.0001]}

save_dir = 'models/lstm'
res_df_lr = hp_tuning(df, param_grid_1, FIRST_TUNING_FILE, device=device, save_dir=save_dir, epochs=MAX_EPOCHS)


def tune_drop_lr(regression):
    with open(FIRST_TUNING_FILE, 'rb') as f:
        res_df = pickle.load(file=f)
    best_params = res_df[res_df['regression'] == regression].sort_values(['mse_mean']).iloc[0,:].to_dict()

    param_grid = {'sequence_length': [best_params['sequence_length']],
                  'no_layers': [best_params['no_layers']],
                  'hidden_dim': [best_params['hidden_dim']],
                  'regression': [regression],
                  'lr': [0.0001],
                  'drop_prob': [0.25, 0.1]}
    model_type = 'regr' if regression else 'clf'
    res_drop_file = f'hp_tuning_{model_type}_drop.pkl'
    _ = hp_tuning(df, param_grid, res_drop_file, device=device, save_dir=save_dir, epochs=MAX_EPOCHS)

    param_grid['drop_prob'] = [0.5]
    param_grid['lr'] = [0.003, 0.001]
    res_lr_file = f'hp_tuning_{model_type}_lr.pkl'
    _ = hp_tuning(df, param_grid, res_lr_file, device=device, save_dir=save_dir, epochs=MAX_EPOCHS)

    return res_drop_file, res_lr_file


res_drop_file_regr, res_lr_file_regr = tune_drop_lr(regression=True)
res_drop_file_clf, res_lr_file_clf = tune_drop_lr(regression=False)


res_dfs = []
for filename in (FIRST_TUNING_FILE, res_drop_file_regr, res_lr_file_regr, res_drop_file_clf, res_lr_file_clf):
    with open(filename, 'rb') as f:
        res_dfs.append(pickle.load(file=f))


res_all = pd.concat(res_dfs, axis=0, ignore_index=True)
with open(SECOND_TUNING_FILE, 'wb') as f:
    pickle.dump(obj=res_all, file=f)