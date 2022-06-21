from src.lstm import get_param_results
import pickle
import matplotlib.pyplot as plt

PARAMS_TUNING_RESULTS_FILE = 'hp_tuning_df.pkl'

with open(PARAMS_TUNING_RESULTS_FILE, 'rb') as f:
    res_df = pickle.load(file=f)
regression = True
res_hidden = get_param_results(res_df, 'hidden_dim', regression)
res_layers = get_param_results(res_df, 'no_layers', regression)
res_seq_len = get_param_results(res_df, 'sequence_length', regression)

fig, ax = plt.subplots(2,3, figsize=(10,8))
ax[0][0].bar(range(3), res_hidden['mse_mean'])
ax[0][0].set_xticks(range(3))
ax[0][0].set_xticklabels(res_hidden['hidden_dim'])
ax[0][0].set_ylabel('MSE Value')
ax[0][0].set_xlabel('LSTM hidden dimension')

ax[0][1].bar(range(3), res_layers['mse_mean'])
ax[0][1].set_xticks(range(3))
ax[0][1].set_xticklabels(res_layers['no_layers'])
ax[0][1].set_xlabel('LSTM number of layers')

ax[0][2].bar(range(3), res_seq_len['mse_mean'])
ax[0][2].set_xticks(range(3))
ax[0][2].set_xticklabels(res_seq_len['sequence_length'])
ax[0][2].set_xlabel('Sequence length')

ax[1][0].bar(range(3), res_hidden['f1_mean'])
ax[1][0].set_xticks(range(3))
ax[1][0].set_xticklabels(res_hidden['hidden_dim'])
ax[1][0].set_ylabel('F1 (macro) Value')
ax[1][0].set_xlabel('LSTM hidden dimension')

ax[1][1].bar(range(3), res_layers['f1_mean'])
ax[1][1].set_xticks(range(3))
ax[1][1].set_xticklabels(res_layers['no_layers'])
ax[1][1].set_xlabel('LSTM number of layers')

ax[1][2].bar(range(3), res_seq_len['f1_mean'])
ax[1][2].set_xticks(range(3))
ax[1][2].set_xticklabels(res_seq_len['sequence_length'])
ax[1][2].set_xlabel('Sequence length')

fig.suptitle('Hyperparameters tuning results')
fig.tight_layout()
fig.savefig('hp_results.png')

