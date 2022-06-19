import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import pandas as pd
from torch.nn.functional import softmax, one_hot
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.data import batch_dataset


class MCDLSTM(nn.Module):
    def __init__(self, no_layers, input_size, hidden_dim, output_dim, drop_prob=0.5, device='cpu'):
        super().__init__()

        self.device = device
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.no_layers = no_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=self.hidden_dim,
                            num_layers=no_layers, batch_first=True, dropout=drop_prob)

        self.dropout = nn.Dropout(drop_prob)

        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x, hidden=None):
        if hidden is None and x.shape[0] > 1:
            lstm_out = torch.zeros((x.shape[0], x.shape[1], self.hidden_dim)).to(self.device)
            for i in range(x.shape[0]):
                single_out, last_hidden = self.lstm(x[i, :, :], hidden)
                lstm_out[i, :, :] = single_out
            hidden = last_hidden
        else:
            lstm_out, hidden = self.lstm(x, hidden)

        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        out = self.fc(out)

        return out, hidden

    # batched input, 0 dimension is batch number
    def estimate_distributions(self, x, n_samples=100):
        self.train()
        batch_size = x.shape[0]
        if self.output_dim == 1:    # regression
            outs = torch.zeros((batch_size, n_samples)).to(self.device)
            for i in range(n_samples):
                out, _ = self.forward(x)
                outs[:, i] = out.squeeze()
            return torch.mean(outs, dim=1).detach().cpu().numpy(), torch.std(outs, dim=1).detach().cpu().numpy()
        else:    # classification
            out_arr = torch.zeros((batch_size, self.output_dim)).to(self.device)
            for i in range(n_samples):
                out, _ = self.forward(x)
                out_oneh = one_hot(torch.argmax(out, dim=1), num_classes=self.output_dim).to(self.device)
                out_arr[:, :] += out_oneh
            return out_arr.cpu().numpy()/n_samples


def train_mcdlstm(model: MCDLSTM, train_dl: DataLoader,
                  dev_dl: DataLoader, epochs, lr, save_prefix,
                  patience=10,
                  print_chart=False, save_chart=False, print_progress=True,
                  device='cpu', bce_weights=None, regression=False,
                  clip_grad=None, progress_bar=True):

    if regression:
        loss_f = nn.MSELoss()
    else:
        loss_f = nn.CrossEntropyLoss(weight=bce_weights)

    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dev_loss_min = np.Inf
    dev_round_mse_min_loss = None
    dev_r2_min_loss = None
    dev_f1_min_loss = None
    dev_report_min_loss = None

    no_improv_epoch_count = 0

    train_losses_epoch = []
    dev_losses_epoch = []
    train_r2s_epoch = []
    dev_r2s_epoch = []

    if progress_bar:
        epoch_range = tqdm(range(epochs))
    else:
        epoch_range = range(epochs)

    for epoch in epoch_range:
        train_losses = []
        train_y = []
        model.train()
        for x, y in train_dl:

            x, y = x.to(device), y.to(device)
            y = y[:, -1]
            if regression:
                y = y.float()

            model.zero_grad()

            output, h = model(x)

            loss = loss_f(output.flatten() if regression else output, y)

            loss.backward()
            train_losses.append(loss.item())

            if len(y) == 1:
                train_y.append(y.squeeze().cpu())
            else:
                train_y.extend(y.squeeze().cpu())

            if clip_grad is not None:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

            optimizer.step()

        dev_losses = []
        dev_y = []
        dev_preds = []

        model.eval()

        for x, y in dev_dl:

            x, y = x.to(device), y.to(device)
            y = y[:, -1]
            if regression:
                y = y.float()

            output, dev_h = model(x)
            dev_loss = loss_f(output.flatten() if regression else output, y)

            dev_losses.append(dev_loss.item())
            if len(y) == 1:
                dev_y.append(y.item())
                if regression:
                    dev_preds.append(output.item())
                else:
                    dev_preds.append(float(torch.argmax(output, dim=1).item()))
            else:
                dev_y.extend(y.squeeze().cpu().numpy())
                if regression:
                    dev_preds.extend(output.detach().cpu().numpy().squeeze())
                else:
                    dev_preds.extend(torch.argmax(output, dim=1).float().cpu().numpy())

        train_loss = np.mean(train_losses)
        train_losses_epoch.append(train_loss)
        dev_loss = np.mean(dev_losses)
        dev_losses_epoch.append(dev_loss)

        train_r2s_epoch.append(1-train_loss/np.var(train_y))
        dev_r2 = 1-dev_loss/np.var(dev_y)
        dev_r2s_epoch.append(dev_r2)

        dev_round_mse = mse_loss(torch.tensor(dev_preds).round(), torch.tensor(dev_y)).item()
        dev_f1 = f1_score(np.rint(dev_y), np.rint(dev_preds), average='macro', zero_division=0)

        dev_report = classification_report(np.rint(dev_y), np.rint(dev_preds), zero_division=0)

        if print_progress:
            print(f'Epoch {epoch + 1}')
            print(f'Train loss : {train_loss}, dev loss : {dev_loss}')
            print(f'Dev MSE (rounded outs) : {dev_round_mse:.3f}, R2: {dev_r2:.3f}, f1: {dev_f1:.3f}')

        if dev_loss <= dev_loss_min:
            dev_round_mse_min_loss = dev_round_mse
            dev_r2_min_loss = dev_r2
            dev_report_min_loss = dev_report
            dev_f1_min_loss = dev_f1
            no_improv_epoch_count = 0
            torch.save(model.state_dict(), save_prefix + '_state_dict.pt')
            dev_loss_min = dev_loss
        else:
            no_improv_epoch_count += 1
        if no_improv_epoch_count == patience:
            print('Early stop.')
            print(f'Best dev loss: {dev_loss_min}')
            break

    best_state = torch.load(save_prefix + '_state_dict.pt')
    model.load_state_dict(best_state)

    if print_chart or save_chart:
        df_loss = pd.DataFrame({'train': train_losses_epoch, 'dev': dev_losses_epoch, })
        df_r2 = pd.DataFrame({'train': train_r2s_epoch, 'dev': dev_r2s_epoch})
        model_type = 'regression' if regression else 'classification'
        ax1 = df_loss.plot.line(title=f'Learning curves for {model_type} model')
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        fig1 = ax1.get_figure()
        ax2 = df_r2.plot.line(title=f'R2 values for {model_type} model')
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('R2')
        fig2 = ax2.get_figure()
        if save_chart:
            fig1.savefig(save_prefix+'_mse.png')
            fig2.savefig(save_prefix+'_r2.png')
        if not print_chart:
            plt.close(fig1)
            plt.close(fig2)

    return dev_round_mse_min_loss, dev_r2_min_loss, dev_f1_min_loss, dev_report_min_loss


def visualise_regr_results(regr_model, x, y, device='cpu'):
    test_ds = TensorDataset(x, y)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    outs = []
    ys = []
    for x, y in test_dl:
        outs.append(np.array(regr_model.estimate_distributions(x.to(device), n_samples=100)).transpose())
        ys.extend(y.numpy()[:,-1])

    outs = np.vstack(outs)
    outs_df = pd.DataFrame({'mean': outs[:,0], 'std': outs[:,1], 'trues': ys})

    plt.plot(outs_df.index, outs_df['mean'], label='pred')
    plt.plot(outs_df.index, outs_df['trues'], label='true')
    plt.fill_between(outs_df.index, y1=outs_df['mean']-outs_df['std'],
                     y2=outs_df['mean'] + outs_df['std'], alpha=.5, color='lightblue', label='1σ range')
    plt.fill_between(outs_df.index, y1=outs_df['mean']-2*outs_df['std'],
                     y2=outs_df['mean'] + 2*outs_df['std'], alpha=.25, color='lightblue', label='2σ range')
    plt.legend()


def visualise_clf_results(clf_model, x, y, device='cpu'):
    test_ds = TensorDataset(x, y)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)
    outs = []
    ys = []
    for x, y in test_dl:
        outs.append(np.array(clf_model.estimate_distributions(x.to(device), n_samples=100)))
        ys.extend(y.numpy()[:,-1])

    outs = np.vstack(outs)

    outs_df = pd.DataFrame({'pred': np.argmax(outs, axis=1),
                            'proba': np.max(outs, axis=1),
                            'true': ys})

    plt.plot(outs_df.index, outs_df['pred'], label='pred')
    plt.plot(outs_df.index, outs_df['true'], label='true')
    plt.bar(outs_df.index, outs_df['proba'], color='gray', alpha=.5, label='pred probability')
    plt.legend()


def evaluate_hparams(data_series, sequence_length=10, no_layers=3,
                     hidden_dim=64, drop_prob=0.5, regression=False,
                     batch_size=32, epochs=200, patience=10, lr=0.00001, silent=False):
    f1_vals = []
    mse_vals = []
    scaler = MinMaxScaler()
    for s in data_series:

        s = (pd.DataFrame(scaler.fit_transform(s[0][0])), s[0][1]), (pd.DataFrame(scaler.transform(s[1][0])), s[1][1])
        (seq_x_train, seq_y_train), (seq_x_test, seq_y_test) = batch_dataset(s, sequence_length=sequence_length, overlap_series=True)

        device = 'cuda:0'

        train_ds = TensorDataset(seq_x_train, seq_y_train)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        test_ds = TensorDataset(seq_x_test, seq_y_test)
        test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

        out_dim = 1 if regression else 4
        mcdLSTM = MCDLSTM(no_layers=no_layers, input_size=12, hidden_dim=hidden_dim, output_dim=out_dim, drop_prob=drop_prob, device=device).to(device)

        save_prefix = 'models/' + ('regr_' if regression else 'clf_') + str(int(time.time()))

        mse, r2, f1, report = train_mcdlstm(mcdLSTM, train_dl, test_dl, regression=regression,
                                            epochs=epochs, lr=lr, patience=patience,
                                            save_prefix=save_prefix, save_chart=True,
                                            print_chart=False, print_progress=False, progress_bar=not silent,
                                            device=device)

        if not silent:
            print(f'MSE: {mse:.3f}, r2: {r2:.3f}, f1 macro: {f1:.3f}')
            print(report)

        f1_vals.append(f1)
        mse_vals.append(mse)
    if not silent:
        print(f'Avg mse: {np.mean(mse_vals)}, f1: {np.mean(f1_vals)}')
    return mse_vals, f1_vals, save_prefix
