import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, classification_report
from torch import nn
import pandas as pd
from torch.nn.functional import softmax, one_hot
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


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
                  patience=10, print_chart=False, print_progress=True,
                  device='cpu', bce_weights=None, regression=False,
                  clip_grad=None):

    if regression:
        loss_f = nn.MSELoss()
    else:
        loss_f = nn.CrossEntropyLoss(weight=bce_weights)

    mse_loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dev_loss_min = np.Inf
    dev_r2_min_loss = None
    dev_f1_min_loss = None
    dev_report_min_loss = None

    no_improv_epoch_count = 0

    train_losses_epoch = []
    dev_losses_epoch = []
    train_r2s_epoch = []
    dev_r2s_epoch = []

    for epoch in tqdm(range(epochs)):
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
                    dev_preds.extend(torch.argmax(output, dim=1).float().cpu().numpy())

        train_loss = np.mean(train_losses)
        train_losses_epoch.append(train_loss)
        dev_loss = np.mean(dev_losses)
        dev_losses_epoch.append(dev_loss)

        train_r2s_epoch.append(1-train_loss/np.var(train_y))
        dev_r2 = 1-dev_loss/np.var(dev_y)
        dev_r2s_epoch.append(dev_r2)

        dev_mse = mse_loss(torch.tensor(dev_preds), torch.tensor(dev_y))
        dev_f1 = f1_score(np.rint(dev_y), np.rint(dev_preds), average='macro', zero_division=0)

        dev_report = classification_report(np.rint(dev_y), np.rint(dev_preds), zero_division=0)

        if print_progress:
            print(f'Epoch {epoch + 1}')
            print(f'Train loss : {train_loss}, dev loss : {dev_loss}')
            print(f'Dev MSE : {dev_mse:.3f}, R2: {dev_r2:.3f}, f1: {dev_f1:.3f}')

        if dev_loss <= dev_loss_min:
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

    if print_chart:
        df_mse = pd.DataFrame({'train_loss': train_losses_epoch, 'dev_loss': dev_losses_epoch, })
        df_r2 = pd.DataFrame({'train_r2': train_r2s_epoch, 'dev_r2': dev_r2s_epoch})
        df_mse.plot.line()
        df_r2.plot.line()

    return dev_loss_min, dev_r2_min_loss, dev_f1_min_loss, dev_report_min_loss


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
                     y2=outs_df['mean'] + outs_df['std'], alpha=.5, color='lightblue')
    plt.fill_between(outs_df.index, y1=outs_df['mean']-2*outs_df['std'],
                     y2=outs_df['mean'] + 2*outs_df['std'], alpha=.25, color='lightblue')
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

    plt.plot(outs_df.index, outs_df['pred'], label='true')
    plt.plot(outs_df.index, outs_df['true'], label='pred')
    plt.bar(outs_df.index, outs_df['proba'], color='gray', alpha=.5, label='pred probability')
    plt.legend()
