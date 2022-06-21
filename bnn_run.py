import numpy as np
import torch
from sklearn.metrics import classification_report

from src.bnn import (
    ELBO,
    BayesianMLP,
    BayesianMLP2,
    fit_elbo,
    show_accuracy_curve,
    show_learning_curve,
)
from src.data import load_data, time_series_split
from src.utils import ensure_reproducibility

df = load_data()


def search_hyperparams(df, exp_name, lr=0.01, hidden_size=64, layers=1, images=False):
    data_series = time_series_split(df, scale='normalize')

    mse = []
    reports = []

    for s in data_series:
        (x_train, y_train), (x_test, y_test) = s

        if layers == 1:
            bnn_model = BayesianMLP(
                num_input_features=12,
                num_hidden_features=hidden_size,
                num_output_classes=4,
            )
        else:
            bnn_model = BayesianMLP2(
                num_input_features=12,
                num_hidden_features=hidden_size,
                num_output_classes=4,
            )

        loss_fun = ELBO(N=10)
        optimizer = torch.optim.Adam(
            bnn_model.parameters(),
            lr=lr,
        )

        train_metrics, test_metrics = fit_elbo(
            model=bnn_model,
            dataset=s,
            loss_function=loss_fun,
            batch_size=32,
            epochs=20,
            optimizer=optimizer,
        )

        preds = bnn_model(torch.tensor(x_test.to_numpy()).to(torch.float32)).argmax(dim=1)
        mse.append(torch.nn.MSELoss()(torch.tensor(y_test.to_numpy()), preds.float()).item())
        reports.append(classification_report(y_test.to_numpy(), preds, zero_division=0))

    print(exp_name, lr, hidden_size)
    print("MSE:", np.mean(mse))
    print(reports[-1])
    if images:
        show_accuracy_curve(train_metrics, test_metrics)
        show_learning_curve(train_metrics, test_metrics)
    torch.save(bnn_model.state_dict(), f'bnn/{exp_name}_lr{lr}_hs{hidden_size}_{layers}.pt')
    return mse, reports


if __name__ == '__main__':
    ensure_reproducibility()

    for lr in [0.01, 0.001, 0.0001]:
        search_hyperparams(df, 'lr_check', lr, 64)
    for hs in [16, 64, 128]:
        search_hyperparams(df, 'hs_check', 0.01, hs)
    for hs in [16, 64, 128]:
        search_hyperparams(df, 'layers_check', 0.01, hs, layers=2)
