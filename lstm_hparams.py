import pickle

import pandas as pd

from src.data import load_data
from src.lstm import hp_tuning

FIRST_TUNING_FILE = 'hp_tuning_df.pkl'
SECOND_TUNING_FILE = 'hp_tuning_all.pkl'

df = load_data()

param_grid_1 = {'sequence_length': [5, 10, 15], 'no_layers': [2, 3, 4], 'hidden_dim': [16,32,64], 'regression': [True, False],
                'drop_prob': [0.5], 'lr': [0.0001]}

#res_df_lr = hp_tuning(df, param_grid_1, FIRST_TUNING_FILE, epochs=1)


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
    res_df_drop = hp_tuning(df, param_grid, f'hp_tuning_{model_type}_drop.pkl')

    param_grid['drop_prob'] = [0.5]
    param_grid['lr'] = [0.003, 0.001]
    res_df_lr = hp_tuning(df, param_grid, f'hp_tuning_{model_type}_lr.pkl')

    return res_df_drop, res_df_lr


res_df_regr_drop, res_df_regr_lr = tune_drop_lr(regression=True)
res_df_clf_drop, res_df_clf_lr = tune_drop_lr(regression=False)

res_all = pd.concat((res_df_regr_drop, res_df_regr_lr, res_df_clf_drop, res_df_clf_lr), axis=0, ignore_index=True)
with open(SECOND_TUNING_FILE, 'wb') as f:
    pickle.dump(obj=res_all, file=f)